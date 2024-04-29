# minds-eye
A One Shot 3D Model Generator

## High Level Runtime Pipeline
- Input: Given an image
- Output: A 3D model of the object in the image

## High Level Training Pipeline
- Given a 3D model, generate n different images from n different view angles
- Train a image-to-images Stable Diffusion model which accepts one of these images and generates an image of the subject from a different view
- Train a images-to-3d-model Stable Diffusion model which accepts images of a subject from n different views and generates a corresponding 3D Model of the subject in the image
