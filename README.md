# PaintingByNumbersIFY
Python code for converting any image to a Painting By Numbers version of itself.

## Usage
```
python pbnify.py --help
usage: pbnify.py [-h] -i INPUT_IMAGE -o OUTPUT_IMAGE [-k NUM_OF_CLUSTERS]
                 [--outline]

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT_IMAGE, --input-image INPUT_IMAGE
                        Path of input image.
  -o OUTPUT_IMAGE, --output-image OUTPUT_IMAGE
                        Path of output image.
  -k NUM_OF_CLUSTERS, --num-of-clusters NUM_OF_CLUSTERS
                        Number of kmeans clusters for dominant color
                        calculation. Defaults to 15.
  --outline             Save outline image containing edges.
```

```
python pbnify.py -i images/picasso.jpg -o images/picasso_PBN.jpg --outline -k 15
```
### Original Image/s:
<img src="https://github.com/CoderHam/PaintingByNumbersIFY/blob/master/images/dancing.jpg" width="400"/> <img src="https://github.com/CoderHam/PaintingByNumbersIFY/blob/master/images/picasso.jpg" width="400"/> <img src="https://github.com/CoderHam/PaintingByNumbersIFY/blob/master/images/hawaii_ham.jpg" width="400"/>

### Image/s converted to their Painting By Number form:
<img src="https://github.com/CoderHam/PaintingByNumbersIFY/blob/master/images/dancing_PBN.jpg" width="400"/> <img src="https://github.com/CoderHam/PaintingByNumbersIFY/blob/master/images/picasso_PBN.jpg" width="400"/> <img src="https://github.com/CoderHam/PaintingByNumbersIFY/blob/master/images/hawaii_ham_PBN.jpg" width="400"/>

### Outline of Image/s converted to their Painting By Number form:
<img src="https://github.com/CoderHam/PaintingByNumbersIFY/blob/master/images/dancing_PBN_outline.jpg" width="400"/> <img src="https://github.com/CoderHam/PaintingByNumbersIFY/blob/master/images/picasso_PBN_outline.jpg" width="400"/> <img src="https://github.com/CoderHam/PaintingByNumbersIFY/blob/master/images/hawaii_ham_PBN_outline.jpg" width="400"/>
