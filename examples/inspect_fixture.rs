use oxideav_dirac::sequence::parse_sequence_header;
use oxideav_dirac::stream::DataUnitIter;
fn main() {
    let data = std::fs::read("tests/fixtures/tiny_1f.drc").unwrap();
    for u in DataUnitIter::new(&data) {
        println!(
            "unit parse_code={:#x} payload_len={}",
            u.parse_info.parse_code,
            u.payload.len()
        );
        if u.parse_info.is_seq_header() {
            let sh = parse_sequence_header(u.payload).unwrap();
            println!("{:#?}", sh);
            println!("raw payload bytes ({}):", u.payload.len());
            for b in u.payload {
                print!("{:02x} ", b);
            }
            println!();
        }
    }
}
