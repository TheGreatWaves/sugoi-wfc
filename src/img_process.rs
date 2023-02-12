use bit_set::BitSet;
use image::{DynamicImage, GenericImage, GenericImageView, ImageBuffer};
use std::{
    collections::{HashMap, HashSet},
    hash::Hash,
    ops::Neg,
};

use ndarray::{ArcArray, ArrayBase, Axis, Dim, Ix2, OwnedArcRepr, Slice};

// Shared two dimensional array
type ArcArray2<A> = ArcArray<A, Ix2>;

// A value indicating the position of a pixel
#[derive(PartialEq, Eq, PartialOrd, Ord, Hash, Copy, Clone, Debug)]
struct Pixel2(u32);

// A 2D section of the image
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Sample2<P>(ArcArray2<P>);

#[derive(PartialEq, Eq, Hash, Clone, Copy, Debug)]
pub struct Offset2(isize, isize);

impl Neg for Offset2 {
    type Output = Self;

    fn neg(self) -> Self {
        Offset2(-self.0, -self.1)
    }
}

// Pixel colour value
type Rgb = image::Rgba<u8>;

pub struct ProcessedInfo {
    _palette: Vec<Rgb>,
    _inverse_palette: HashMap<Rgb, Pixel2>,

    _samples: Vec<Sample2<Pixel2>>,
    _weights: Vec<(usize, f64)>,

    _collide: HashMap<Offset2, Vec<BitSet>>,

    _n: usize,
}

/**
 * Create a palette set (all unique colours present in the image).
 */
fn create_palette_set(img: &DynamicImage) -> HashSet<Rgb> {
    // Note: .map captures (x, y, rgb)
    img.pixels().map(|(_, _, rgb)| rgb).collect()
}

/**
 * Map each colour in the palette to an identifer.
 */
fn create_palette_map(palette_set: HashSet<Rgb>) -> HashMap<Rgb, Pixel2> {
    palette_set
        .into_iter()
        .enumerate()
        .map(|(i, p)| (p, Pixel2(i as u32)))
        .collect()
}

/**
 * Copy an image into a buffer.
 * This is used for further construction.
 */
fn create_img_buffer(img: &DynamicImage) -> ImageBuffer<Rgb, Vec<u8>> {
    let mut buf = ImageBuffer::<Rgb, _>::new(img.width(), img.height());
    _ = buf.copy_from(img, 0, 0);
    buf
}

/**
 * Create a palette ArcArray.
 */
fn create_palette_arcarr(
    img: &DynamicImage,
    palette_map: &HashMap<Rgb, Pixel2>,
) -> ArcArray2<Pixel2> {
    let buf = create_img_buffer(img);

    let shape = (buf.width() as usize, buf.height() as usize);
    let get = |(x, y)| palette_map[buf.get_pixel(x as u32 % buf.width(), y as u32 % buf.height())];

    ArcArray::from_shape_fn(shape, get)
}

/**
 * Sample the image.
 *
 * We can use the samples and weights to help the WFC algorithm to
 * produce results which are more accurate to the input image. This
 * can be done by altering the probabilities of getting a specific
 * sample given its weight.
 */
fn sample(
    palette_array: ArrayBase<OwnedArcRepr<Pixel2>, Dim<[usize; 2]>>,
    section_count: &usize,
) -> (Vec<Sample2<Pixel2>>, Vec<(usize, f64)>) {
    let mut sample_set = HashMap::new();

    // In each
    for i in 0..palette_array.dim().0 - (section_count - 1) {
        for j in 0..palette_array.dim().1 - (section_count - 1) {
            // Note: Marked mutable for slice_axis_inplace
            let mut sample = palette_array.to_shared();

            // Select the ranges of target rows (Note: this creates a view so no data is modified here)
            sample.slice_axis_inplace(
                Axis(0),
                Slice::from(i as isize..(i + section_count) as isize),
            );

            // Filter to retrieve only the columns we want
            sample.slice_axis_inplace(
                Axis(1),
                Slice::from(j as isize..(j + section_count) as isize),
            );

            // Add new entry or increment if existing
            *sample_set.entry(Sample2(sample)).or_insert(0) += 1;
        }
    }

    // Upzip the map for keys (samples) and values (weight)
    let (sample_vec, weight_vec): (Vec<_>, Vec<_>) = sample_set.into_iter().unzip();

    // Assign an index to each weight (The index will correspond perfectly to the sample the weight belongs to)
    let weights: Vec<_> = weight_vec
        .into_iter()
        .enumerate()
        .map(|(i, x)| (i, x as f64))
        .collect();

    (sample_vec, weights)
}

/**
 * Generates collision map
 */
fn generate_collision_map(
    samples: &Vec<Sample2<Pixel2>>,
    section_count: &usize,
) -> HashMap<Offset2, Vec<BitSet>> {
    let mut collide = HashMap::new();

    /*
     * Check whether two samples can seamlessly join.
     *
     * dx: Length of the region
     * dy: Height of the region
     * lx: Starting x position on the left sample
     * ly: Starting y position on the left sample
     * rx: Starting x position on the right sample
     * ry: Starting y position on the right sample
     */
    let check_at_offset = |region_length, region_height, lx, ly, rx, ry| {
        let mut bitsets = Vec::new();
        for Sample2(left) in samples.iter() {
            // Create a bitset which helps us indicate whether
            // a given sample is compatible with another.
            let mut bs = BitSet::with_capacity(samples.len());

            'others: for (s, Sample2(right)) in samples.iter().enumerate() {
                // Check the region of sample 'right' WRT sample 'left'
                // If the region is equivalent it means that they are compatible.
                for i in 0..region_length {
                    for j in 0..region_height {
                        let p_l = left[((lx + i), (ly + j))];
                        let p_r = right[((rx + i), (ry + j))];
                        if p_l != p_r {
                            // Mismatch, process the next one.
                            continue 'others;
                        }
                    }
                }
                // Compatible
                bs.insert(s);
            }
            bitsets.push(bs);
        }
        bitsets
    };

    let n: usize = *section_count;
    for dx in 0..n {
        for dy in 0..n {
            let o_dx = dx as isize;
            let o_dy = dy as isize;

            let region_length = section_count - dx;
            let region_height = section_count - dy;

            // Check for same orientation
            collide.insert(
                Offset2(o_dx, o_dy),
                check_at_offset(region_length, region_height, dx, dy, 0, 0),
            );

            // Check for 90 deg rotation
            collide.insert(
                Offset2(-o_dx, o_dy),
                check_at_offset(region_length, region_height, 0, dy, dx, 0),
            );

            // Check for 180 deg rotation
            collide.insert(
                Offset2(o_dx, -o_dy),
                check_at_offset(region_length, region_height, dx, 0, 0, dy),
            );

            // Check for 270 deg rotation
            collide.insert(
                Offset2(-o_dx, -o_dy),
                check_at_offset(region_length, region_height, 0, 0, dx, dy),
            );
        }
    }

    collide
}

/**
 * Creates a vector of the colours in the palette, sorted by their pixel position.
 */
fn create_palette(palette_map: &HashMap<Rgb, Pixel2>) -> Vec<Rgb> {
    let mut vec: Vec<_> = palette_map.iter().map(|(&rgb, &px)| (rgb, px)).collect();
    vec.sort_by_key(|x| x.1);
    vec.into_iter().map(|x| x.0).collect()
}

/**
 * This function takes in an img and the section_size which represents (section_count x section_count)
 */
pub fn process_img(img: &DynamicImage, section_count: usize) -> ProcessedInfo {
    let palette_set: HashSet<Rgb> = create_palette_set(img);
    let palette_map: HashMap<Rgb, Pixel2> = create_palette_map(palette_set);
    let palette_array: ArcArray2<Pixel2> = create_palette_arcarr(img, &palette_map);

    let (samples, weights) = sample(palette_array, &section_count);

    let collision_map = generate_collision_map(&samples, &section_count);

    let palette = create_palette(&palette_map);

    ProcessedInfo {
        _palette: palette,
        _inverse_palette: palette_map,
        _samples: samples,
        _weights: weights,
        _collide: collision_map,
        _n: section_count,
    }
}
