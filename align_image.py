import cv2
import numpy as np
import argparse
import os


def pad_to_align(image, align=64):
    """Pad image to be divisible by align value."""
    h, w = image.shape[:2]
    
    # Calculate padded size
    new_w = (w + align - 1) // align * align
    new_h = (h + align - 1) // align * align
    
    # Pad with zeros (black) on right and bottom
    if len(image.shape) == 3:
        padded = np.zeros((new_h, new_w, image.shape[2]), dtype=image.dtype)
    else:
        padded = np.zeros((new_h, new_w), dtype=image.dtype)
    
    padded[:h, :w] = image
    
    print(f"Original: {w}x{h} -> Padded: {new_w}x{new_h}")
    return padded, (w, h)


def crop_to_original(image, original_size):
    """Crop image back to original size."""
    w, h = original_size
    return image[:h, :w]


def main():
    parser = argparse.ArgumentParser(description='Pad image to aligned size and crop output back')
    parser.add_argument('-i', '--input', required=True, help='Input image path')
    parser.add_argument('-o', '--output', default='output_cropped.png', help='Output cropped image path')
    parser.add_argument('--align', type=int, default=64, help='Alignment value (default: 64)')
    parser.add_argument('--padded', default='input_padded.png', help='Padded input image path')
    parser.add_argument('--sdk-output', help='SDK output image to crop (if provided)')
    args = parser.parse_args()
    
    # Read input image
    img = cv2.imread(args.input)
    if img is None:
        print(f"Error: Cannot read image {args.input}")
        return
    
    # Pad to aligned size
    padded_img, original_size = pad_to_align(img, args.align)
    
    # Save padded image for SDK processing
    cv2.imwrite(args.padded, padded_img)
    print(f"Saved padded image to: {args.padded}")
    
    # If SDK output is provided, crop it back
    if args.sdk_output and os.path.exists(args.sdk_output):
        sdk_out = cv2.imread(args.sdk_output)
        if sdk_out is not None:
            cropped = crop_to_original(sdk_out, original_size)
            cv2.imwrite(args.output, cropped)
            print(f"Saved cropped output to: {args.output}")
        else:
            print(f"Error: Cannot read SDK output {args.sdk_output}")
    else:
        print(f"\nNext steps:")
        print(f"1. Run SDK with: {args.padded}")
        print(f"2. Then run: python align_image.py -i {args.input} --sdk-output <sdk_output.png> -o {args.output}")


if __name__ == '__main__':
    main()
