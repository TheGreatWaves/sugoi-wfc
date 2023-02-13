use image::{DynamicImage, GenericImageView};

use crate::data::{
    colour::{self, make_rgb, Rgb},
    vector2::Vector2,
};

#[allow(dead_code)]
#[derive(Debug)]
pub struct Image {
    pub width: usize,
    pub height: usize,
    pub pixels: Vec<Rgb>,
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

    pub fn idx(&self, at: Vector2) -> usize {
        (at.y as usize * self.width) + at.x as usize
    }

    #[allow(dead_code)]
    pub fn at(&self, at: Vector2) -> Rgb {
        let idx = self.idx(at);
        self.pixels[idx]
    }

    pub fn set_colour(&mut self, at: Vector2, colour: Rgb) {
        let idx = self.idx(at);
        self.pixels[idx] = colour;
    }

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
}
