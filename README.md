# legofy
Build pictures with LEGO®

### Install
Just get all packages with (make sure that you are using Python 3):

```bash
pip3 install -r requirements.txt
```

### Usage
The `legofy.py` script can be executed via
```bash
python3 legofy.py [options]
```

The options should be printed if you call `python3 legofy.py -h`

```
>>> python3 legofy.py -h
usage: legofy.py [-h] -i INPUT [-o OUTPUT] [-sm SMOOTH] [-s SIZE]
                 [-ms MAXSIZE]

Legofy your image.

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
  -s SIZE, --size SIZE  Max size for the output image in pixels/studs.
                        Defaults to 32.
  -ms MAXSIZE, --maxsize MAXSIZE
                        Max size of an individual LEGO plate in studs.
                        Defaults to 12.
```

Have fun legofying around!
