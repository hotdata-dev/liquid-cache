use std::mem::size_of;
use std::sync::LazyLock;

use arrow::array::{Array, ArrayRef, BooleanArray};
use arrow::buffer::{BooleanBuffer, NullBuffer};
use arrow_schema::{DataType, Field};
use bytes::Bytes;
use datafusion_common::ScalarValue;
use datafusion_expr_common::operator::Operator as DataFusionOperator;
use datafusion_physical_expr::expressions::{BinaryExpr, Column, LikeExpr, Literal};
use vortex_array::arrays::ConstantArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::scalar_fn::fns::like::{Like, LikeOptions};
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_array::serde::{SerializeOptions, SerializedArray};
use vortex_array::{ArrayContext, ArrayId, IntoArray, VortexSessionExecute};
use vortex_arrow::ArrowSessionExt;
use vortex_btrblocks::BtrBlocksCompressor;
use vortex_buffer::{Alignment, BitBuffer, ByteBuffer, ByteBufferMut};
use vortex_error::VortexResult;
use vortex_fsst::FSST;
use vortex_ipc::messages::{BufMessageReader, DecoderMessage, EncoderMessage, MessageEncoder};
use vortex_mask::Mask;
use vortex_session::VortexSession;
use vortex_session::registry::ReadContext;

use crate::cache::LiquidExpr;

use super::eval_predicate_on_array;

const MAGIC: u32 = 0x4C51_4441; // "LQDA" for LiQuid Data Array
const VERSION: u16 = 2;
const HEADER_SIZE: usize = 16;
/// Header byte holding the exponent of the alignment a block is written for.
const ALIGNMENT_OFFSET: usize = 6;
/// The least alignment a serialized block is written for.
const MIN_ALIGNMENT: Alignment = Alignment::new(64);

static SESSION: LazyLock<VortexSession> = LazyLock::new(session);
static COMPRESSOR: LazyLock<BtrBlocksCompressor> = LazyLock::new(BtrBlocksCompressor::default);

fn session() -> VortexSession {
    let session = vortex_array::array_session();
    vortex_arrow::initialize(&session);
    vortex_fastlanes::initialize(&session);
    vortex_fsst::initialize(&session);
    vortex_alp::initialize(&session);
    vortex_runend::initialize(&session);
    vortex_sparse::initialize(&session);
    vortex_zigzag::initialize(&session);
    vortex_datetime_parts::initialize(&session);
    vortex_decimal_byte_parts::initialize(&session);
    vortex_bytebool::initialize(&session);
    vortex_sequence::initialize(&session);
    session
}

/// A Liquid array backed by an in-memory compressed Vortex array.
#[derive(Debug)]
pub struct LiquidArray {
    array: vortex_array::ArrayRef,
    arrow_type: DataType,
    /// What [`Self::get_array_memory_size`] reports; see [`heap_size`].
    memory_size: usize,
}

impl LiquidArray {
    /// Transcode a supported Arrow array into Vortex.
    pub fn from_arrow_array(array: &ArrayRef) -> Result<Self, &ArrayRef> {
        Self::compress(array).map(Self::compacted)
    }

    /// The compressor's output, which can still share memory with `array`
    /// and carry spare capacity; see [`Self::compacted`].
    fn compress(array: &ArrayRef) -> Result<Self, &ArrayRef> {
        if !is_supported(array.data_type()) {
            return Err(array);
        }

        let arrow_type = array.data_type().clone();
        let imported = match import_array(array) {
            Ok(imported) => imported,
            Err(error) => {
                log::warn!("failed to import {arrow_type:?} into Vortex: {error}");
                return Err(array);
            }
        };
        let mut ctx = SESSION.create_execution_ctx();
        let array = match COMPRESSOR.compress(&imported, &mut ctx) {
            Ok(compressed) => compressed,
            Err(error) => {
                log::warn!("failed to compress {arrow_type:?} with Vortex: {error}");
                return Err(array);
            }
        };
        // Only the compacted array is ever charged; this one is transient.
        let memory_size = array.nbytes() as usize + size_of::<Self>();
        Ok(Self {
            array,
            arrow_type,
            memory_size,
        })
    }

    /// Detach the array from its source's buffers and from the compressor's
    /// spare capacity.
    ///
    /// The cache charges an entry by `nbytes()`, the sum of its buffers'
    /// lengths, but the compressor's output can keep more alive: a constant
    /// string is a zero-copy slice of its whole source buffer (vortex #10599),
    /// and an FSST codes buffer keeps its growth slack (vortex #11420). Both
    /// are freed only with the entry. Round-tripping through the
    /// serialization the disk tier uses drops those allocations, and leaves
    /// an array whose heap [`heap_size`] can account for.
    fn compacted(self) -> Self {
        let (alignment, pieces) = self.encode();
        let len = pieces.iter().map(ByteBuffer::len).sum();
        let mut block = ByteBufferMut::with_capacity_aligned(len, alignment);
        for piece in &pieces {
            block.extend_from_slice(piece);
        }
        drop((self, pieces));
        Self::decode(block.freeze())
    }

    /// Deserialize a Vortex-backed Liquid array.
    pub fn from_bytes(bytes: Bytes) -> Self {
        Self::decode(ByteBuffer::from(bytes))
    }

    /// Decode a serialized block, keeping every buffer of the result a
    /// zero-copy slice of it.
    ///
    /// The decoder copies out any buffer it finds misaligned, while the block
    /// stays alive for the others, so the copied bytes would be held twice.
    /// [`Self::encode`] pads every buffer to its alignment relative to the
    /// block's start; this aligns the start itself, copying the block once
    /// if the caller's allocation is not aligned enough.
    fn decode(block: ByteBuffer) -> Self {
        validate_header(&block);
        let alignment = Alignment::try_from_untrusted_exponent(block[ALIGNMENT_OFFSET])
            .expect("valid Liquid array alignment");
        // Align the start, then drop the requirement so sections slice freely.
        let block = block.aligned(alignment).aligned(Alignment::none());

        let mut cursor = HEADER_SIZE;
        let mut next_section = || {
            let start = cursor + 4;
            cursor = start + read_u32(&block, cursor) as usize;
            start..cursor
        };
        let arrow_type = next_section();
        let dtype_message = next_section();
        let encodings = next_section();
        let row_count = read_u32(&block, cursor) as usize;
        let segment_start = (cursor + 4).next_multiple_of(alignment.as_usize());

        let arrow_type =
            serde_json::from_slice(&block[arrow_type]).expect("valid serialized Arrow data type");
        let encodings: Vec<String> =
            serde_json::from_slice(&block[encodings]).expect("valid serialized encoding ids");
        let encodings: Vec<ArrayId> = encodings.iter().map(|id| ArrayId::new(id)).collect();
        let mut reader = BufMessageReader::new(block.slice(dtype_message).into_bytes());
        let dtype = match reader
            .next()
            .expect("Vortex dtype message")
            .expect("valid Vortex IPC")
        {
            DecoderMessage::DType(dtype) => {
                DType::from_flatbuffer(dtype, &SESSION).expect("valid Vortex dtype")
            }
            message => panic!("expected Vortex dtype message, got {message:?}"),
        };
        let array = SerializedArray::try_from(block.slice(segment_start..))
            .expect("valid Vortex array segment")
            .decode(&dtype, row_count, &ReadContext::new(encodings), &SESSION)
            .expect("valid Vortex array");
        let memory_size = heap_size(&array, &block) + size_of::<Self>();
        Self {
            array,
            arrow_type,
            memory_size,
        }
    }

    /// The serialized form, as pieces to write contiguously into a block
    /// aligned to the returned alignment.
    ///
    /// Layout: the header, three length-prefixed sections (the Arrow type as
    /// JSON, the Vortex dtype as an IPC message, the array's encoding ids as
    /// JSON), the row count, and the array segment. The segment starts at a
    /// multiple of the alignment and pads each buffer to its own alignment,
    /// so every buffer decodes in place.
    fn encode(&self) -> (Alignment, Vec<ByteBuffer>) {
        let dtype_message = MessageEncoder::new(SESSION.clone())
            .encode(EncoderMessage::DType(self.array.dtype()))
            .expect("Vortex dtype serialization")
            .concat();
        let context = ArrayContext::empty();
        let segment = self
            .array
            .serialize(
                &context,
                &SESSION,
                &SerializeOptions {
                    offset: 0,
                    include_padding: true,
                },
            )
            .expect("Vortex array serialization");
        let encodings: Vec<String> = context
            .to_ids()
            .iter()
            .map(|id| id.as_ref().to_string())
            .collect();
        // The segment's first piece is empty and carries the largest
        // alignment its buffers ask for.
        let alignment = segment
            .iter()
            .map(ByteBuffer::alignment)
            .fold(MIN_ALIGNMENT, Ord::max);

        let mut head = header_bytes(alignment).to_vec();
        for section in [
            serde_json::to_vec(&self.arrow_type).expect("Arrow data type serializes"),
            dtype_message,
            serde_json::to_vec(&encodings).expect("encoding ids serialize"),
        ] {
            let len = u32::try_from(section.len()).expect("Liquid array section fits in u32");
            head.extend_from_slice(&len.to_le_bytes());
            head.extend_from_slice(&section);
        }
        let row_count = u32::try_from(self.array.len()).expect("Liquid array length fits in u32");
        head.extend_from_slice(&row_count.to_le_bytes());
        head.resize(head.len().next_multiple_of(alignment.as_usize()), 0);

        let mut pieces = Vec::with_capacity(segment.len() + 1);
        pieces.push(ByteBuffer::from(head));
        pieces.extend(segment);
        (alignment, pieces)
    }

    fn export(&self, array: vortex_array::ArrayRef) -> ArrayRef {
        let field = Field::new("", self.arrow_type.clone(), array.dtype().is_nullable());
        let mut ctx = SESSION.create_execution_ctx();
        SESSION
            .arrow()
            .execute_arrow(array, Some(&field), &mut ctx)
            .expect("Vortex array must export to its import Arrow type")
    }

    fn filtered_vortex(&self, filter: &BooleanBuffer) -> vortex_array::ArrayRef {
        assert_eq!(filter.len(), self.len(), "filter length must match array");
        let mask = Mask::from_buffer(BitBuffer::from(filter.clone()));
        self.array.filter(mask).expect("Vortex filter must succeed")
    }

    fn try_vortex_predicate(
        &self,
        array: &vortex_array::ArrayRef,
        predicate: &LiquidExpr,
    ) -> Option<vortex_array::ArrayRef> {
        let expr = predicate.physical_expr();
        if let Some(binary) = expr.downcast_ref::<BinaryExpr>() {
            binary.left().downcast_ref::<Column>()?;
            let literal = binary.right().downcast_ref::<Literal>()?;
            return match binary.op() {
                DataFusionOperator::Eq
                | DataFusionOperator::NotEq
                | DataFusionOperator::Lt
                | DataFusionOperator::LtEq
                | DataFusionOperator::Gt
                | DataFusionOperator::GtEq => {
                    let op = vortex_operator(binary.op())?;
                    let constant = self.constant(literal.value(), array.len())?;
                    let array = if matches!(op, Operator::Eq | Operator::NotEq) {
                        with_private_fsst_symbol_tables(array).ok()?
                    } else {
                        array.clone()
                    };
                    array.binary(constant, op).ok()
                }
                DataFusionOperator::LikeMatch | DataFusionOperator::NotLikeMatch => self.try_like(
                    array,
                    literal.value(),
                    LikeOptions {
                        negated: matches!(binary.op(), DataFusionOperator::NotLikeMatch),
                        case_insensitive: false,
                    },
                ),
                _ => None,
            };
        }

        if let Some(like) = expr.downcast_ref::<LikeExpr>() {
            if like.case_insensitive() || like.expr().downcast_ref::<Column>().is_none() {
                return None;
            }
            let pattern = like.pattern().downcast_ref::<Literal>()?;
            return self.try_like(
                array,
                pattern.value(),
                LikeOptions {
                    negated: like.negated(),
                    case_insensitive: like.case_insensitive(),
                },
            );
        }
        None
    }

    fn constant(&self, value: &ScalarValue, len: usize) -> Option<vortex_array::ArrayRef> {
        let value = value.cast_to(&self.arrow_type).ok()?;
        let arrow = value.to_array_of_size(1).ok()?;
        let vortex = SESSION
            .arrow()
            .from_arrow_array(arrow, value.is_null())
            .ok()?;
        let mut ctx = SESSION.create_execution_ctx();
        let scalar = vortex.execute_scalar(0, &mut ctx).ok()?;
        Some(ConstantArray::new(scalar, len).into_array())
    }

    fn try_like(
        &self,
        array: &vortex_array::ArrayRef,
        pattern: &ScalarValue,
        options: LikeOptions,
    ) -> Option<vortex_array::ArrayRef> {
        let pattern = self.constant(pattern, array.len())?;
        Some(
            Like::try_new(array.clone(), pattern, options)
                .ok()?
                .into_array(),
        )
    }

    fn try_eval_vortex_predicate(
        &self,
        array: &vortex_array::ArrayRef,
        predicate: &LiquidExpr,
    ) -> Option<BooleanArray> {
        let result = self.try_vortex_predicate(array, predicate)?;
        let field = Field::new("", DataType::Boolean, result.dtype().is_nullable());
        let mut ctx = SESSION.create_execution_ctx();
        let result = SESSION
            .arrow()
            .execute_arrow(result, Some(&field), &mut ctx)
            .ok()?;
        let result = result.as_any().downcast_ref::<BooleanArray>()?;
        let source_validity = array
            .validity()
            .ok()?
            .execute_mask(array.len(), &mut ctx)
            .ok()?;
        let source_nulls = vortex_arrow::to_null_buffer(source_validity);
        // Vortex compare on extension dtypes returns non-null false for null input rows.
        // Liquid's contract is null where the input is null (see Timestamp = literal).
        let nulls = NullBuffer::union(result.nulls(), source_nulls.as_ref());
        Some(BooleanArray::new(result.values().clone(), nulls))
    }
}

impl LiquidArray {
    /// Whether any node of the encoded array is FSST-compressed strings.
    #[doc(hidden)]
    pub fn uses_fsst(&self) -> bool {
        self.array
            .depth_first_traversal()
            .any(|node| node.is::<FSST>())
    }

    /// Get the memory size of the Liquid array.
    pub fn get_array_memory_size(&self) -> usize {
        self.memory_size
    }

    /// Get the length of the Liquid array.
    pub fn len(&self) -> usize {
        self.array.len()
    }

    /// Check whether the Liquid array is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Convert the Liquid array to an Arrow array.
    pub fn to_arrow_array(&self) -> ArrayRef {
        self.export(self.array.clone())
    }

    /// Get the original Arrow data type.
    pub fn original_arrow_data_type(&self) -> DataType {
        self.arrow_type.clone()
    }

    /// Serialize the Liquid array.
    pub fn to_bytes(&self) -> Vec<u8> {
        let (_, pieces) = self.encode();
        let mut bytes = Vec::with_capacity(pieces.iter().map(ByteBuffer::len).sum());
        for piece in &pieces {
            bytes.extend_from_slice(piece);
        }
        bytes
    }

    /// Filter the Liquid array and return an Arrow array.
    pub fn filter(&self, selection: &BooleanBuffer) -> ArrayRef {
        self.export(self.filtered_vortex(selection))
    }

    /// Evaluate a predicate on selected rows.
    pub fn try_eval_predicate(
        &self,
        predicate: &LiquidExpr,
        filter: &BooleanBuffer,
    ) -> BooleanArray {
        let filtered = self.filtered_vortex(filter);
        if let Some(value) = predicate
            .physical_expr()
            .downcast_ref::<Literal>()
            .and_then(|literal| match literal.value() {
                ScalarValue::Boolean(value) => *value,
                _ => None,
            })
        {
            let values = if value {
                BooleanBuffer::new_set(filtered.len())
            } else {
                BooleanBuffer::new_unset(filtered.len())
            };
            let mut ctx = SESSION.create_execution_ctx();
            let validity = filtered
                .validity()
                .and_then(|validity| validity.execute_mask(filtered.len(), &mut ctx))
                .expect("Vortex array validity must execute");
            return BooleanArray::new(values, vortex_arrow::to_null_buffer(validity));
        }
        let Some(result) = self.try_eval_vortex_predicate(&filtered, predicate) else {
            return eval_predicate_on_array(self.export(filtered), predicate);
        };
        result
    }
}

fn header_bytes(alignment: Alignment) -> [u8; HEADER_SIZE] {
    let mut bytes = [0; HEADER_SIZE];
    bytes[0..4].copy_from_slice(&MAGIC.to_le_bytes());
    bytes[4..6].copy_from_slice(&VERSION.to_le_bytes());
    bytes[ALIGNMENT_OFFSET] = alignment.exponent();
    bytes
}

/// Heap each node of a decoded Vortex tree holds besides its buffers: the
/// shared array allocation, its statistics set, slots and dtype. Measured
/// with a counting allocator (`tests/liquid_array_memory.rs`).
///
/// Both constants depend on Vortex's internals and can drift when it is
/// upgraded; `charge_matches_the_heap_an_entry_holds` then fails. To
/// re-derive them, run `cargo test -p liquid-cache --test liquid_array_memory
/// -- --nocapture`, which prints each fixture's charge and real heap, and
/// adjust until the ratios sit around 1.0: `NODE_HEAP` from the fixtures with
/// the most nodes per byte, then `FSST_SYMBOL_TABLE_HEAP` from the FSST ones.
const NODE_HEAP: usize = 640;
/// Heap an FSST node holds besides that: the decoder re-pads its symbol
/// table to the full 256 entries of 8-byte symbols and 1-byte lengths, each
/// in an allocation of its own, outside the block.
const FSST_SYMBOL_TABLE_HEAP: usize = 4800;

/// The heap a decoded array holds: the block, if any buffer still points
/// into it, every buffer that does not, and per-node overhead.
///
/// `nbytes()`, the sum of buffer lengths, leaves out the block's framing and
/// padding and all per-node heap. For a well-compressed column that is most
/// of what the entry holds: a few hundred bytes of buffers can sit under
/// several kilobytes of nodes.
fn heap_size(array: &vortex_array::ArrayRef, block: &ByteBuffer) -> usize {
    let block_start = block.as_ptr() as usize;
    let block_range = block_start..block_start + block.len();
    let mut block_held = false;
    let mut size = 0;
    for node in array.depth_first_traversal() {
        size += NODE_HEAP;
        let fsst = node.is::<FSST>();
        if fsst {
            size += FSST_SYMBOL_TABLE_HEAP;
        }
        for (index, buffer) in node.buffers().iter().enumerate() {
            let start = buffer.as_ptr() as usize;
            if !buffer.is_empty()
                && block_range.contains(&start)
                && start + buffer.len() <= block_range.end
            {
                block_held = true;
            } else if !(fsst && index < 2) {
                // FSST's first two buffers are the symbol table, counted
                // above at its padded size.
                size += buffer.len();
            }
        }
    }
    if block_held {
        size += block.len();
    }
    size
}

/// `array` with every FSST node given a symbol table of its own.
///
/// FSST evaluates `=` and `!=` by compressing the literal, and memoizes the
/// compressor it builds for that (about 160 KiB) in the node's symbol table.
/// Slices and filters share that table with the cached entry, so the first
/// such predicate would grow the entry for the rest of its life, and nothing
/// recharges it. With a private table, the compressor is freed with the
/// predicate's array. Only the table is copied (2.3 KiB); the codes are not.
fn with_private_fsst_symbol_tables(
    array: &vortex_array::ArrayRef,
) -> VortexResult<vortex_array::ArrayRef> {
    if !array.depth_first_traversal().any(|node| node.is::<FSST>()) {
        return Ok(array.clone());
    }
    let slots = array
        .slots()
        .iter()
        .map(|slot| {
            slot.as_ref()
                .map(with_private_fsst_symbol_tables)
                .transpose()
        })
        .collect::<VortexResult<vortex_array::ArraySlots>>()?;
    // SAFETY: every slot is replaced by the same values with, at most, a new
    // FSST symbol table holding the same symbols.
    let array = unsafe { array.clone().with_slots(slots)? };
    if !array.is::<FSST>() {
        return Ok(array);
    }
    // Rebuilding an FSST node from its own buffers builds a new symbol table.
    // SAFETY: the buffers are the node's own.
    let buffers = array.buffer_handles();
    unsafe { array.with_buffers(buffers) }
}

fn read_u32(bytes: &[u8], at: usize) -> u32 {
    let field = bytes
        .get(at..at + 4)
        .expect("serialized Liquid array is truncated");
    u32::from_le_bytes(field.try_into().expect("four bytes"))
}

fn validate_header(bytes: &[u8]) {
    assert!(
        bytes.len() >= HEADER_SIZE,
        "value too small for Liquid array header, expected at least {HEADER_SIZE} bytes, got {}",
        bytes.len()
    );
    let magic = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
    let version = u16::from_le_bytes(bytes[4..6].try_into().unwrap());
    assert_eq!(magic, MAGIC, "Invalid Liquid array magic number");
    assert_eq!(version, VERSION, "Unsupported Liquid array version");
}

fn import_array(array: &ArrayRef) -> VortexResult<vortex_array::ArrayRef> {
    SESSION
        .arrow()
        .from_arrow_array(array.clone(), array.nulls().is_some())
}

fn is_supported(data_type: &DataType) -> bool {
    matches!(
        data_type,
        DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Date32
            | DataType::Date64
            | DataType::Timestamp(_, None)
            | DataType::Float32
            | DataType::Float64
            | DataType::Decimal128(_, _)
            | DataType::Decimal256(_, _)
            | DataType::Utf8
            | DataType::Utf8View
            | DataType::Binary
            | DataType::BinaryView
    ) || matches!(
        data_type,
        DataType::Dictionary(key, value)
            if key.as_ref() == &DataType::UInt16
                && matches!(value.as_ref(), DataType::Utf8 | DataType::Binary)
    )
}

fn vortex_operator(op: &DataFusionOperator) -> Option<Operator> {
    match op {
        DataFusionOperator::Eq => Some(Operator::Eq),
        DataFusionOperator::NotEq => Some(Operator::NotEq),
        DataFusionOperator::Lt => Some(Operator::Lt),
        DataFusionOperator::LtEq => Some(Operator::Lte),
        DataFusionOperator::Gt => Some(Operator::Gt),
        DataFusionOperator::GtEq => Some(Operator::Gte),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::{
        BinaryArray, BinaryViewArray, BooleanArray, Date32Array, Decimal128Array, DictionaryArray,
        Float64Array, Int32Array, Int64Array, StringArray, StringViewArray, StructArray,
        TimestampMicrosecondArray, UInt16Array,
    };
    use arrow::datatypes::UInt16Type;
    use datafusion_physical_expr::PhysicalExpr;
    use datafusion_physical_expr::expressions::CastExpr;

    use super::*;

    fn roundtrip(array: ArrayRef) {
        let vortex = LiquidArray::from_arrow_array(&array).unwrap();
        assert_eq!(vortex.to_arrow_array().as_ref(), array.as_ref());
    }

    #[test]
    fn roundtrips_supported_arrow_types() {
        let decimal = Decimal128Array::from(vec![Some(12_345), None, Some(-9_876)])
            .with_precision_and_scale(15, 3)
            .unwrap();
        let dictionary = DictionaryArray::<UInt16Type>::from_iter([
            Some("red"),
            None,
            Some("blue"),
            Some("red"),
        ]);
        let arrays: Vec<ArrayRef> = vec![
            Arc::new(Int32Array::from_iter_values(0..1_100)),
            Arc::new(Int64Array::from(vec![Some(-1), None, Some(7)])),
            Arc::new(UInt16Array::from(vec![0, 1, u16::MAX])),
            Arc::new(Date32Array::from(vec![Some(0), None, Some(20_000)])),
            Arc::new(TimestampMicrosecondArray::from(vec![
                Some(1_000),
                None,
                Some(9_000),
            ])),
            Arc::new(Float64Array::from(vec![Some(1.5), None, Some(-0.0)])),
            Arc::new(decimal),
            Arc::new(StringArray::from(vec![Some("alpha"), None, Some("")])),
            Arc::new(StringViewArray::from(vec![
                Some(""),
                None,
                Some("a longer string value"),
            ])),
            Arc::new(BinaryArray::from(vec![
                Some(b"valid utf8".as_ref()),
                None,
                Some(b"".as_ref()),
            ])),
            Arc::new(BinaryViewArray::from_iter_values([
                vec![0xff, 0x00],
                vec![],
                vec![0x80],
            ])),
            Arc::new(dictionary),
        ];
        for array in arrays {
            roundtrip(array);
        }
    }

    #[test]
    fn roundtrips_empty_array() {
        roundtrip(Arc::new(StringViewArray::from(Vec::<Option<&str>>::new())));
    }

    #[test]
    fn rejects_unsupported_types() {
        let boolean: ArrayRef = Arc::new(BooleanArray::from(vec![true, false]));
        assert!(LiquidArray::from_arrow_array(&boolean).is_err());

        let values: ArrayRef = Arc::new(Int32Array::from(vec![1, 2]));
        let structure: ArrayRef = Arc::new(StructArray::from(vec![(
            Arc::new(Field::new("x", DataType::Int32, false)),
            values,
        )]));
        assert!(LiquidArray::from_arrow_array(&structure).is_err());
    }

    #[test]
    fn filters_like_arrow() {
        let array: ArrayRef = Arc::new(Int64Array::from(vec![Some(1), None, Some(3), Some(4)]));
        let selection = BooleanBuffer::from(vec![true, false, true, false]);
        let vortex = LiquidArray::from_arrow_array(&array).unwrap();
        let expected =
            arrow::compute::filter(&array, &BooleanArray::new(selection.clone(), None)).unwrap();
        assert_eq!(vortex.filter(&selection).as_ref(), expected.as_ref());
    }

    fn comparison_expr(
        op: DataFusionOperator,
        value: ScalarValue,
        data_type: &DataType,
    ) -> LiquidExpr {
        let expr: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(Column::new("c", 0)),
            op,
            Arc::new(Literal::new(value)),
        ));
        LiquidExpr::try_new(expr, data_type).unwrap()
    }

    fn like_expr(pattern: ScalarValue, negated: bool, data_type: &DataType) -> LiquidExpr {
        let expr: Arc<dyn PhysicalExpr> = Arc::new(LikeExpr::new(
            negated,
            false,
            Arc::new(Column::new("c", 0)),
            Arc::new(Literal::new(pattern)),
        ));
        LiquidExpr::try_new(expr, data_type).unwrap()
    }

    fn expected_predicate(
        array: &ArrayRef,
        expr: &LiquidExpr,
        selection: &BooleanBuffer,
    ) -> BooleanArray {
        let filtered =
            arrow::compute::filter(array, &BooleanArray::new(selection.clone(), None)).unwrap();
        eval_predicate_on_array(filtered, expr)
    }

    fn assert_predicate_matches(name: &str, array: &ArrayRef, expr: &LiquidExpr) {
        let selection = BooleanBuffer::new_set(array.len());
        let vortex = LiquidArray::from_arrow_array(array).unwrap();
        let filtered = vortex.filtered_vortex(&selection);
        if vortex.try_eval_vortex_predicate(&filtered, expr).is_none() {
            eprintln!("Vortex predicate fell back to Arrow: {name}");
        }
        assert_eq!(
            vortex.try_eval_predicate(expr, &selection),
            expected_predicate(array, expr, &selection),
            "predicate case {name}"
        );
    }

    #[test]
    fn evaluates_comparisons_with_vortex() {
        let integers: ArrayRef = Arc::new(Int64Array::from(vec![
            Some(-2),
            None,
            Some(0),
            Some(3),
            Some(9),
        ]));
        let strings: ArrayRef = Arc::new(StringViewArray::from(vec![
            Some("x"),
            None,
            Some("prefix-x"),
            Some("z"),
            Some("x-suffix"),
        ]));
        let selections = [
            BooleanBuffer::new_set(5),
            BooleanBuffer::from(vec![true, false, true, false, true]),
        ];
        let ops = [
            DataFusionOperator::Eq,
            DataFusionOperator::NotEq,
            DataFusionOperator::Lt,
            DataFusionOperator::LtEq,
            DataFusionOperator::Gt,
            DataFusionOperator::GtEq,
        ];

        for selection in selections {
            let vortex = LiquidArray::from_arrow_array(&integers).unwrap();
            for op in ops {
                for literal in [-1, 3] {
                    let expr = comparison_expr(
                        op,
                        ScalarValue::Int64(Some(literal)),
                        integers.data_type(),
                    );
                    assert_eq!(
                        vortex.try_eval_predicate(&expr, &selection),
                        expected_predicate(&integers, &expr, &selection)
                    );
                }
            }

            let vortex = LiquidArray::from_arrow_array(&strings).unwrap();
            for op in ops {
                for literal in ["x", "z"] {
                    let expr = comparison_expr(
                        op,
                        ScalarValue::Utf8View(Some(literal.into())),
                        strings.data_type(),
                    );
                    assert_eq!(
                        vortex.try_eval_predicate(&expr, &selection),
                        expected_predicate(&strings, &expr, &selection)
                    );
                }
            }
        }
    }

    #[test]
    fn evaluates_tricky_predicates() {
        let dictionary: ArrayRef = Arc::new(DictionaryArray::<UInt16Type>::from_iter([
            Some("x"),
            None,
            Some("prefix-x"),
            Some("z"),
        ]));
        for op in [
            DataFusionOperator::Eq,
            DataFusionOperator::NotEq,
            DataFusionOperator::Lt,
        ] {
            let name = format!("Dictionary(UInt16, Utf8) {op} Utf8");
            let expr = comparison_expr(
                op,
                ScalarValue::Utf8(Some("x".into())),
                dictionary.data_type(),
            );
            assert_predicate_matches(&name, &dictionary, &expr);
        }
        let expr = like_expr(
            ScalarValue::Utf8(Some("%x%".into())),
            false,
            dictionary.data_type(),
        );
        assert_predicate_matches("Dictionary(UInt16, Utf8) LIKE Utf8", &dictionary, &expr);

        let binary: ArrayRef = Arc::new(BinaryArray::from(vec![
            Some(b"ax".as_slice()),
            None,
            Some(b"by".as_slice()),
            Some(b"cz".as_slice()),
        ]));
        for op in [DataFusionOperator::Eq, DataFusionOperator::Lt] {
            let name = format!("Binary {op} Binary");
            let expr = comparison_expr(
                op,
                ScalarValue::Binary(Some(b"by".to_vec())),
                binary.data_type(),
            );
            assert_predicate_matches(&name, &binary, &expr);
        }

        let binary_view: ArrayRef = Arc::new(BinaryViewArray::from(vec![
            Some(b"\xff\x00".as_slice()),
            None,
            Some(b"\x80".as_slice()),
            Some(b"valid".as_slice()),
        ]));
        let expr = comparison_expr(
            DataFusionOperator::Eq,
            ScalarValue::BinaryView(Some(b"\x80".to_vec())),
            binary_view.data_type(),
        );
        assert_predicate_matches("invalid BinaryView = BinaryView", &binary_view, &expr);

        let dates: ArrayRef = Arc::new(Date32Array::from(vec![Some(0), None, Some(20_000)]));
        for (op, value) in [
            (DataFusionOperator::Gt, 10_000),
            (DataFusionOperator::Eq, 20_000),
        ] {
            let name = format!("Date32 {op} Date32");
            let expr = comparison_expr(op, ScalarValue::Date32(Some(value)), dates.data_type());
            assert_predicate_matches(&name, &dates, &expr);
        }

        let timestamps: ArrayRef = Arc::new(TimestampMicrosecondArray::from(vec![
            Some(1_000),
            None,
            Some(9_000),
        ]));
        for (op, value) in [
            (DataFusionOperator::Gt, 5_000),
            (DataFusionOperator::Eq, 9_000),
        ] {
            let name = format!("Timestamp(Microsecond) {op} Timestamp(Microsecond)");
            let expr = comparison_expr(
                op,
                ScalarValue::TimestampMicrosecond(Some(value), None),
                timestamps.data_type(),
            );
            assert_predicate_matches(&name, &timestamps, &expr);
        }

        let decimals: ArrayRef = Arc::new(
            Decimal128Array::from(vec![Some(1_000), None, Some(12_345)])
                .with_precision_and_scale(15, 3)
                .unwrap(),
        );
        let expr = comparison_expr(
            DataFusionOperator::GtEq,
            ScalarValue::Decimal128(Some(5_000), 15, 3),
            decimals.data_type(),
        );
        assert_predicate_matches("Decimal128(15, 3) >= Decimal128", &decimals, &expr);

        let integers: ArrayRef = Arc::new(Int64Array::from(vec![Some(1), None, Some(3)]));
        let physical: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(CastExpr::new(
                Arc::new(Column::new("c", 0)),
                DataType::Int32,
                None,
            )),
            DataFusionOperator::Eq,
            Arc::new(Literal::new(ScalarValue::Int32(Some(3)))),
        ));
        let expr = LiquidExpr::try_new(physical, integers.data_type()).unwrap();
        assert_predicate_matches("Int64 = Int32", &integers, &expr);

        let strings: ArrayRef = Arc::new(StringViewArray::from(vec![
            Some("alpha"),
            None,
            Some("beta"),
        ]));
        let physical: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(CastExpr::new(
                Arc::new(Column::new("c", 0)),
                DataType::Utf8,
                None,
            )),
            DataFusionOperator::Eq,
            Arc::new(Literal::new(ScalarValue::Utf8(Some("beta".into())))),
        ));
        let expr = LiquidExpr::try_new(physical, strings.data_type()).unwrap();
        assert_predicate_matches("Utf8View = Utf8", &strings, &expr);
    }

    #[test]
    fn evaluates_like_with_vortex() {
        let array: ArrayRef = Arc::new(StringViewArray::from(vec![
            Some("x"),
            None,
            Some("prefix-x"),
            Some("x-suffix"),
            Some("none"),
        ]));
        let vortex = LiquidArray::from_arrow_array(&array).unwrap();
        for (pattern, negated) in [("%x%", false), ("x%", false), ("%x", false), ("%x%", true)] {
            let expr = like_expr(
                ScalarValue::Utf8View(Some(pattern.into())),
                negated,
                &DataType::Utf8View,
            );
            let selection = BooleanBuffer::new_set(array.len());
            assert_eq!(
                vortex.try_eval_predicate(&expr, &selection),
                expected_predicate(&array, &expr, &selection)
            );
        }
    }

    #[test]
    fn boolean_literal_preserves_string_nulls() {
        let strings = (0..3_000)
            .map(|index| (index % 11 != 0).then_some("a nullable string value"))
            .collect::<Vec<_>>();
        let array: ArrayRef = Arc::new(StringViewArray::from(strings));
        let physical: Arc<dyn PhysicalExpr> =
            Arc::new(Literal::new(ScalarValue::Boolean(Some(true))));
        let predicate = LiquidExpr::try_new(physical, array.data_type()).unwrap();
        let vortex = LiquidArray::from_arrow_array(&array).unwrap();
        assert_eq!(
            vortex.try_eval_predicate(&predicate, &BooleanBuffer::new_set(array.len())),
            BooleanArray::new(BooleanBuffer::new_set(array.len()), array.nulls().cloned())
        );
    }

    #[test]
    fn roundtrips_dictionary_export_branches() {
        let identical_values = vec!["same"; 64];
        let identical: ArrayRef = Arc::new(DictionaryArray::<UInt16Type>::from_iter(
            identical_values.iter().map(|value| Some(*value)),
        ));

        let unique_values = (0..256)
            .map(|index| format!("unique-{index}"))
            .collect::<Vec<_>>();
        let unique: ArrayRef = Arc::new(DictionaryArray::<UInt16Type>::from_iter(
            unique_values.iter().map(|value| Some(value.as_str())),
        ));

        let long_values = (0..1_500)
            .map(|index| format!("value-{}", index % 7))
            .collect::<Vec<_>>();
        let long: ArrayRef = Arc::new(DictionaryArray::<UInt16Type>::from_iter(
            long_values.iter().map(|value| Some(value.as_str())),
        ));

        for array in [identical, unique, long] {
            roundtrip(array);
        }
    }

    #[test]
    fn ipc_roundtrips() {
        let arrays: Vec<ArrayRef> = vec![
            Arc::new(Int64Array::from(vec![Some(1), None, Some(3)])),
            Arc::new(StringViewArray::from(vec![
                Some("one"),
                None,
                Some("three"),
            ])),
        ];
        for array in arrays {
            let vortex = LiquidArray::from_arrow_array(&array).unwrap();
            let decoded = LiquidArray::from_bytes(Bytes::from(vortex.to_bytes()));
            assert_eq!(decoded.to_arrow_array().as_ref(), array.as_ref());
        }
    }

    #[test]
    #[should_panic(expected = "Invalid Liquid array magic number")]
    fn rejects_bad_ipc_magic() {
        LiquidArray::from_bytes(Bytes::from_static(&[0; HEADER_SIZE]));
    }

    #[test]
    #[should_panic(expected = "Unsupported Liquid array version")]
    fn rejects_bad_ipc_version() {
        let mut header = header_bytes(MIN_ALIGNMENT);
        header[4..6].copy_from_slice(&(VERSION + 1).to_le_bytes());
        LiquidArray::from_bytes(Bytes::copy_from_slice(&header));
    }

    /// A column of one repeated string compresses to a constant. The entry
    /// must not keep the source column's buffers alive: the cache charges
    /// it only for the constant, while the source buffers are 8192 rows wide.
    #[test]
    fn constant_strings_do_not_pin_the_source_buffers() {
        for value in ["user", "a value longer than the inline twelve bytes"] {
            let source = StringViewArray::from_iter_values(std::iter::repeat_n(value, 8192));
            let views = source.views().inner().clone();
            let data = source.data_buffers().to_vec();
            let array: ArrayRef = Arc::new(source);

            let liquid = LiquidArray::from_arrow_array(&array).unwrap();
            drop(array);

            assert!(
                liquid.array.as_constant().is_some(),
                "{value:?} should compress to a constant"
            );
            assert!(
                views.into_mutable().is_ok(),
                "{value:?}: the entry still holds the source views"
            );
            for buffer in data {
                assert!(
                    buffer.into_mutable().is_ok(),
                    "{value:?}: the entry still holds a source data buffer"
                );
            }
            assert_eq!(
                liquid.to_arrow_array().as_ref(),
                &StringViewArray::from_iter_values(std::iter::repeat_n(value, 8192)) as &dyn Array
            );
        }
    }

    /// An FSST codes buffer grows by doubling while it is built and keeps
    /// that spare capacity, which `nbytes()` does not see. The compacted
    /// entry must not share that allocation.
    #[test]
    fn fsst_codes_growth_allocation_is_released() {
        let mut x: u64 = 1;
        let strings: Vec<String> = (0..8192)
            .map(|_| {
                (0..8)
                    .map(|_| {
                        x ^= x << 13;
                        x ^= x >> 7;
                        x ^= x << 17;
                        format!("{x:016x}")
                    })
                    .collect()
            })
            .collect();
        let array: ArrayRef = Arc::new(StringViewArray::from_iter_values(&strings));

        let compressed = LiquidArray::compress(&array).unwrap();
        let fsst = compressed
            .array
            .depth_first_traversal()
            .find(|node| node.encoding_id().as_ref() == "vortex.fsst")
            .expect("random hex strings should compress with FSST");
        let codes = fsst
            .buffers()
            .into_iter()
            .max_by_key(|buffer| buffer.len())
            .unwrap();
        drop(fsst);

        let liquid = compressed.compacted();
        let codes = codes
            .try_into_mut()
            .expect("the entry still holds the compressor's codes allocation");
        assert!(
            codes.capacity() > codes.len() * 3 / 2,
            "the fixture should leave growth slack: len {} capacity {}",
            codes.len(),
            codes.capacity()
        );
        assert_eq!(liquid.to_arrow_array().as_ref(), array.as_ref());
    }

    /// Columns whose encoded buffers land at offsets their alignment rejects
    /// unless the serializer pads for them.
    fn misalignment_prone_columns() -> Vec<(&'static str, ArrayRef)> {
        let mut state: u64 = 1;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let binary = BinaryArray::from_iter_values((0..8192).map(|_| {
            let mut value = next().to_le_bytes().to_vec();
            value.extend_from_slice(&next().to_le_bytes());
            value
        }));
        let timestamps = TimestampMicrosecondArray::from_iter_values(
            (0..8192).map(|row| 1_700_000_000_000_000 + row * 1000 + (next() % 997) as i64),
        );
        let hex = StringArray::from_iter_values((0..8192).map(|_| format!("{:016x}", next())));
        let urls = StringArray::from_iter_values(
            (0..8192)
                .map(|row| format!("https://example.com/path/1/{row}/item?id={}", next() >> 33)),
        );
        vec![
            ("Binary random 16 B", Arc::new(binary)),
            ("Timestamp(us) ms ramp", Arc::new(timestamps)),
            ("Utf8 hex16", Arc::new(hex)),
            ("Utf8 urls", Arc::new(urls)),
        ]
    }

    /// Bytes of `liquid`'s data buffers that live outside `block`.
    ///
    /// Two kinds of buffer are never in the block and are left out: a
    /// constant's scalar, which `buffers()` serializes on demand, and an FSST
    /// symbol table, which the decoder re-pads into a table of its own.
    fn bytes_outside(liquid: &LiquidArray, block: std::ops::Range<usize>) -> Vec<String> {
        let mut outside = Vec::new();
        for node in liquid.array.depth_first_traversal() {
            let encoding = node.encoding_id().as_ref().to_string();
            let names = node.buffer_names();
            for (index, buffer) in node.buffers().into_iter().enumerate() {
                let name = names.get(index).map(String::as_str).unwrap_or("?");
                if buffer.is_empty()
                    || encoding == "vortex.constant"
                    || (encoding == "vortex.fsst" && name != "compressed_codes")
                {
                    continue;
                }
                let start = buffer.as_slice().as_ptr() as usize;
                if start < block.start || start + buffer.len() > block.end {
                    outside.push(format!("{encoding}.{name} ({} bytes)", buffer.len()));
                }
            }
        }
        outside
    }

    /// A Liquid entry is decoded from one serialized block and keeps it
    /// alive. A buffer the decoder copies out of that block, because it sits
    /// at an offset its alignment rejects, is then held twice: once in the
    /// block and once in the copy.
    #[test]
    fn decode_keeps_every_buffer_in_the_serialized_block() {
        for (name, column) in misalignment_prone_columns() {
            let liquid = LiquidArray::from_arrow_array(&column).unwrap();
            // An allocation aligned for any buffer, so only the layout
            // inside the block decides whether a buffer is copied.
            let block = vortex_buffer::ByteBufferMut::copy_from_aligned(
                liquid.to_bytes(),
                vortex_buffer::Alignment::new(64),
            )
            .freeze()
            .into_bytes();
            let start = block.as_ptr() as usize;
            let decoded = LiquidArray::from_bytes(block.clone());
            let outside = bytes_outside(&decoded, start..start + block.len());
            assert!(
                outside.is_empty(),
                "{name}: decoding copied buffers out of the block: {outside:?}"
            );
            assert_eq!(decoded.to_arrow_array().as_ref(), column.as_ref());
        }
    }
}
