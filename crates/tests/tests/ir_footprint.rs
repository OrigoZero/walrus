//! What the IR holds for each instruction it parses, and the DWARF
//! bookkeeping it records only for a module whose DWARF it rewrites.

use walrus::ir::{dfs_in_order, Instr, InstrSeq, Value, Visitor};
use walrus::{InstrLocId, Module, ModuleConfig};

const WAT: &str = r#"
(module
  (func (export "f") (param i32) (result i32)
    (block (result i32)
      local.get 0
      i32.const 1
      i32.add
      (if (result i32) (local.get 0) (then (i32.const 2)) (else (i32.const 3)))
      i32.add)
    (loop $l (br_if $l (i32.const 0)))
    v128.const i64x2 1 2
    drop)
  (func (export "g") (result i64)
    i64.const 7))
"#;

/// The module `WAT` describes, with an empty `.debug_info` section after it
/// when `with_dwarf` is set.
fn module_bytes(with_dwarf: bool) -> Vec<u8> {
    let mut bytes = wat::parse_str(WAT).unwrap();
    if with_dwarf {
        let name = b".debug_info";
        bytes.push(0); // custom section id
        bytes.push((1 + name.len()) as u8); // section size
        bytes.push(name.len() as u8);
        bytes.extend_from_slice(name);
    }
    bytes
}

fn parse(with_dwarf: bool, generate_dwarf: bool) -> Module {
    let mut config = ModuleConfig::new();
    config.generate_dwarf(generate_dwarf);
    config.parse(&module_bytes(with_dwarf)).unwrap()
}

/// Every instruction sequence of every local function.
fn for_each_seq(module: &Module, mut f: impl FnMut(&InstrSeq)) {
    struct Seqs<'a, F: FnMut(&InstrSeq)>(&'a mut F);
    impl<'instr, F: FnMut(&InstrSeq)> Visitor<'instr> for Seqs<'_, F> {
        fn start_instr_seq(&mut self, seq: &'instr InstrSeq) {
            (self.0)(seq);
        }
    }
    for (_, func) in module.funcs.iter_local() {
        dfs_in_order(&mut Seqs(&mut f), func, func.entry_block());
    }
}

#[test]
fn an_instruction_and_its_location_take_at_most_fifty_six_bytes() {
    // Every parsed instruction is one `(Instr, InstrLocId)`, so this size is
    // the IR's cost per instruction of the input: a 42 MB code section of
    // 16.8 million instructions holds 900 MiB of them at 56 bytes.
    let size = std::mem::size_of::<(Instr, InstrLocId)>();
    assert!(
        size <= 56,
        "(Instr, InstrLocId) is {size} bytes; the IR holds one per parsed instruction and is \
         budgeted at 56. Box the payload of the rare variant that grew it."
    );
    assert_eq!(std::mem::align_of::<Instr>(), 8);
}

#[test]
fn a_parsed_sequence_holds_no_spare_capacity() {
    for with_dwarf in [false, true] {
        let module = parse(with_dwarf, true);
        let mut seqs = 0;
        for_each_seq(&module, |seq| {
            seqs += 1;
            assert_eq!(
                seq.instrs.capacity(),
                seq.instrs.len(),
                "a parsed sequence keeps the room its instructions take"
            );
        });
        assert!(
            seqs >= 5,
            "the module's blocks, if arms and loop are walked"
        );
    }
}

#[test]
fn a_v128_constant_keeps_its_bytes_through_parse_and_emit() {
    let expected = 1u128 | (2u128 << 64);
    let find = |module: &Module| {
        let mut found = None;
        for_each_seq(module, |seq| {
            for (instr, _) in seq.instrs.iter() {
                if let Instr::Const(c) = instr {
                    if let Value::V128(bytes) = c.value {
                        found = Some(bytes);
                    }
                }
            }
        });
        found.expect("the v128.const is parsed")
    };
    let mut module = parse(false, false);
    assert_eq!(find(&module), expected.to_le_bytes());
    let value = Value::V128(expected.to_le_bytes());
    assert_eq!(value.to_string(), expected.to_string());

    let emitted = module.emit_wasm();
    let reparsed = ModuleConfig::new().parse(&emitted).unwrap();
    assert_eq!(find(&reparsed), expected.to_le_bytes());
}
