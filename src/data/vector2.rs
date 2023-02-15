use std::ops::Sub;

#[allow(dead_code)]
pub struct Vector2 {
    pub x: i32,
    pub y: i32,
}

impl Sub for Vector2 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Vector2 {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
        }
    }
}
