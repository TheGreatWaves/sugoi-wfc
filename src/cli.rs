use clap::Parser;

#[derive(Parser, Default, Debug)]
pub struct Args {
    // Name of the image file
    pub img_path: String,
    pub n_dimensions: usize,
}
