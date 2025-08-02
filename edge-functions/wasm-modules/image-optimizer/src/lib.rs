use wasm_bindgen::prelude::*;
use image::{DynamicImage, ImageFormat, ImageOutputFormat};
use std::io::Cursor;

#[wasm_bindgen]
pub struct ImageOptimizer {
    quality: u8,
    max_width: u32,
    max_height: u32,
}

#[wasm_bindgen]
impl ImageOptimizer {
    #[wasm_bindgen(constructor)]
    pub fn new(quality: u8, max_width: u32, max_height: u32) -> Self {
        Self {
            quality,
            max_width,
            max_height,
        }
    }

    #[wasm_bindgen]
    pub fn optimize(&self, image_data: &[u8], format: &str) -> Result<Vec<u8>, JsValue> {
        // Parse image
        let img = image::load_from_memory(image_data)
            .map_err(|e| JsValue::from_str(&format!("Failed to load image: {}", e)))?;

        // Resize if necessary
        let img = self.resize_if_needed(img);

        // Convert to WebP for optimal compression
        let mut output = Vec::new();
        let mut cursor = Cursor::new(&mut output);
        
        match format {
            "webp" => {
                img.write_to(&mut cursor, ImageOutputFormat::WebP)
                    .map_err(|e| JsValue::from_str(&format!("Failed to encode WebP: {}", e)))?;
            }
            "jpeg" | "jpg" => {
                img.write_to(&mut cursor, ImageOutputFormat::Jpeg(self.quality))
                    .map_err(|e| JsValue::from_str(&format!("Failed to encode JPEG: {}", e)))?;
            }
            _ => {
                img.write_to(&mut cursor, ImageOutputFormat::Png)
                    .map_err(|e| JsValue::from_str(&format!("Failed to encode PNG: {}", e)))?;
            }
        }

        Ok(output)
    }

    #[wasm_bindgen]
    pub fn generate_thumbnail(&self, image_data: &[u8], width: u32, height: u32) -> Result<Vec<u8>, JsValue> {
        let img = image::load_from_memory(image_data)
            .map_err(|e| JsValue::from_str(&format!("Failed to load image: {}", e)))?;

        // Generate thumbnail with smart cropping
        let thumbnail = img.resize_to_fill(width, height, image::imageops::FilterType::Lanczos3);

        // Output as WebP for best compression
        let mut output = Vec::new();
        let mut cursor = Cursor::new(&mut output);
        thumbnail.write_to(&mut cursor, ImageOutputFormat::WebP)
            .map_err(|e| JsValue::from_str(&format!("Failed to encode thumbnail: {}", e)))?;

        Ok(output)
    }

    #[wasm_bindgen]
    pub fn extract_dominant_colors(&self, image_data: &[u8], num_colors: u32) -> Result<Vec<u32>, JsValue> {
        let img = image::load_from_memory(image_data)
            .map_err(|e| JsValue::from_str(&format!("Failed to load image: {}", e)))?;

        // Simple k-means clustering for dominant colors
        let pixels: Vec<[u8; 3]> = img.to_rgb8()
            .pixels()
            .map(|p| [p[0], p[1], p[2]])
            .collect();

        // Simplified color extraction (in production, use proper k-means)
        let mut colors = Vec::new();
        let step = pixels.len() / num_colors as usize;
        
        for i in 0..num_colors as usize {
            let idx = i * step;
            if idx < pixels.len() {
                let p = &pixels[idx];
                let color = ((p[0] as u32) << 16) | ((p[1] as u32) << 8) | (p[2] as u32);
                colors.push(color);
            }
        }

        Ok(colors)
    }

    fn resize_if_needed(&self, img: DynamicImage) -> DynamicImage {
        let (width, height) = img.dimensions();
        
        if width > self.max_width || height > self.max_height {
            let ratio = (self.max_width as f32 / width as f32)
                .min(self.max_height as f32 / height as f32);
            
            let new_width = (width as f32 * ratio) as u32;
            let new_height = (height as f32 * ratio) as u32;
            
            img.resize(new_width, new_height, image::imageops::FilterType::Lanczos3)
        } else {
            img
        }
    }
}

// Edge worker entry point
#[wasm_bindgen]
pub async fn handle_image_request(
    url: String,
    width: Option<u32>,
    height: Option<u32>,
    quality: Option<u8>,
    format: Option<String>,
) -> Result<Vec<u8>, JsValue> {
    let optimizer = ImageOptimizer::new(
        quality.unwrap_or(85),
        width.unwrap_or(1920),
        height.unwrap_or(1080),
    );

    // In a real implementation, fetch the image from URL
    // For now, assume image_data is passed
    let image_data = vec![]; // Placeholder
    
    let output_format = format.as_deref().unwrap_or("webp");
    optimizer.optimize(&image_data, output_format)
}