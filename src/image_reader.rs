use image::{DynamicImage, GenericImageView};

use crate::data::{
    colour::{self, make_rgb, Rgb},
    vector2::Vector2,
};

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct Image {
    pub width: usize,
    pub height: usize,
    pub pixels: Vec<Rgb>,
}

#[derive(Debug)]
pub struct Sample {
    pub region: Vec<Rgb>,
}

impl Sample {
    pub fn new() -> Sample {
        Sample { region: Vec::new() }
    }
}

impl Image {
    /**
     * Initialize a new image buffer with given width and height.
     */
    pub fn new(width: usize, height: usize) -> Image {
        Image {
            width,
            height,
            pixels: vec![colour::BLACK; width * height],
        }
    }

    /**
     * Retrieve index position from Vector2 position
     */
    pub fn idx(&self, at: Vector2) -> usize {
        (at.y as usize * self.width) + at.x as usize
    }

    /**
     * Retrieve colour from Vector2 position
     */
    pub fn at(&self, at: Vector2) -> Rgb {
        let idx = self.idx(at);
        self.pixels[idx]
    }

    /**
     * Set colour at Vector2 position
     */
    pub fn set_colour(&mut self, at: Vector2, colour: Rgb) {
        let idx = self.idx(at);
        self.pixels[idx] = colour;
    }

    /**
     * Load and save an image
     */
    pub fn load(&mut self, img: &DynamicImage) {
        img.pixels().for_each(|(x, y, rgb)| {
            self.set_colour(
                Vector2 {
                    x: x as i32,
                    y: y as i32,
                },
                make_rgb(&rgb),
            );
        });
    }

    /**
     * Slice a sample from the loaded image
     *
     * xs:      Starting x position
     * ys:      Starting y position
     * width:   Region width
     * height:  Region height
     */
    pub fn get_region(&self, xs: &i32, ys: &i32, width: &i32, height: &i32) -> Sample {
        let mut sample: Sample = Sample::new();
        for i in *xs..(*xs + width) {
            for j in *ys..(*ys + height) {
                let i = i as usize;
                let j = j as usize;
                sample.region.push(self.at(Vector2 {
                    x: i as i32 % self.width as i32,
                    y: j as i32 % self.height as i32,
                }))
            }
        }
        sample
    }

    /**
     * Sample the image
     */
    pub fn sample(&self, n: i32) -> Vec<Sample> {
        let sampler = |xs, ys| self.get_region(&xs, &ys, &n, &n);
        self.pixels
            .iter()
            .enumerate()
            .fold(Vec::<Sample>::new(), |mut sample_vec, (idx, _)| {
                let x = idx % self.width;
                let y = idx / self.width;
                sample_vec.push(sampler(x as i32, y as i32));
                sample_vec
            })
    }
}
