# Pic2Brick
Build pictures with LEGO® or any other compatible bricks.

![There should be an image here.](logo.png)

### Install
Just get all packages with (make sure that you are using Python 3):

```bash
pip3 install -r requirements.txt
```

### Usage
The `pic2brick.py` script can be executed via
```bash
python3 pic2brick.py [options]
```

The options should be printed if you call `python3 pic2brick.py -h`

```
>>> python3 pic2brick.py -h
usage: pic2brick.py [-h] -i INPUT [-o OUTPUT] [-sm SMOOTH] [-s SIZE]
                 [-ms MAXSIZE] [-rgb RGB]

Build your image with bricks.

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        input image
  -o OUTPUT, --output OUTPUT
                        output xml file. Defaults to out.xml and overwrites
                        any previous output with the same name.
  -sm SMOOTH, --smooth SMOOTH
                        Smoothing factor for prefiltering. Increase for
                        removing artifacts. Can only be odd. Defaults to 1.
  -s SIZE, --size SIZE  
                        Max size for the output image in pixels/studs.
                        Defaults to 32.
  -ml MAXLENGTH, --maxlength MAXLENGTH
                        Max length of an individual LEGO plate in studs.
                        Defaults to 12.
  -mw MAXWIDTH, --maxwidth MAXWIDTH
                        Max width of an individual LEGO plate in studs.
                        Defaults to 6.
  -lab LAB              
                        If not zero, LAB is used for distances between pixels,
                        otherwise RGB. Defaults to 1.
```

Have fun bricking around!

LEGO® is a registered trademark of the LEGO group of companies and does not sponsor, authorize, or endorse this project.
