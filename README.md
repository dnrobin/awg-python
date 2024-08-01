# awg-python v1.0.0

Arrayed waveguide grating (AWG) simulator for nanophotonics designed as a series of functional blocks in an object oriented architecture. See the example notebooks to learn how to use the library step-by-step in this order: basics, functions, highlevel and advanced.

## Installation

### Dependencies

The dependancies can be found in the `requirements.txt` file. Workable with python3.11 (Tested ✅)  
In order to get the environement work as standalone, please create your own python environement and then install the requirements. The code will be ready to play 
```shell
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.standalone.txt
```

In order to get the environement work within your code base, please make sure the following dependencies are installed, and compatible with your current environement:
```shell
pip install -r requirements.txt
```

### Instructions

In order to use this library, unzip and copy the 'awg' sub folder into your local directory and simply import the module.

```python
import awg
```

## Contributing

Please file any issues encountered to: github.com/dnrobin/awg-python/issues

To contribute to the source code:

* Fork a copy of the project
* Clone that copy to your local machine
* Make edits/contributions
* Commit and push changes to your branch
* Create a pull request on the 'master' branch

## Version history

### Version 1.0.0 (Feb 22, 2021)
* initial port of the `awg-matlab` library to python