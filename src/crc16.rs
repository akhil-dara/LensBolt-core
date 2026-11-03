/// CRC-16/CCITT-FALSE: poly 0x1021, init 0xFFFF, no reflection, no final xor.
pub fn crc16(data: &[u8]) -> u16 {
    let mut crc: u16 = 0xFFFF;
    for &b in data {
        crc ^= (b as u16) << 8;
        for _ in 0..8 {
            crc = if crc & 0x8000 != 0 { (crc << 1) ^ 0x1021 } else { crc << 1 };
        }
    }
    crc
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn check_value_matches_ccitt_false() {
        assert_eq!(crc16(b"123456789"), 0x29B1);
    }
    #[test]
    fn empty_is_init() {
        assert_eq!(crc16(b""), 0xFFFF);
    }
}
