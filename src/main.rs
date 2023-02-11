use clap::Parser;
mod img_process;

#[derive(Parser, Default, Debug)]
struct Args {
    // Name of the image file
    img_path: String,
    n_dimensions: usize,
}

fn main() {
    // Parse CLI
    let args = Args::parse();

    // Load image from args passed in
    let img = image::open(args.img_path).expect("Failed to open image");

    // Process the image
    img_process::process_img(&img, args.n_dimensions);
}
