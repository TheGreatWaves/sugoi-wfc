use clap::Parser;
use cli::Args;
extern crate image;

mod cli;
mod data;
mod image_reader;

fn main() {
    // Parse CLI
    let args = Args::parse();

    // Load image from args passed in
    let img = image::open(args.img_path).expect("Failed to open image");

    // Process the image
    let mut image = image_reader::Image::new(img.width() as usize, img.height() as usize);
    image.load(&img);

    let sample = image.sample(args.n_dimensions as i32);
    dbg!(sample);
}
