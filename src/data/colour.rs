use image::Rgba;

#[allow(dead_code)]
pub type Rgb = [u8; 3];

#[allow(dead_code)]
pub const BLACK: Rgb = [0, 0, 0];

#[allow(dead_code)]
pub fn make_rgb(rgb: &Rgba<u8>) -> Rgb {
    rgb.0[0..3].try_into().expect("RGB: Incorrect format")
}
