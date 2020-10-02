# AIGEN FRVT 1:1 Archive Repository

**This repository is an archive to keep track of progress**


> Download [model.tar.gz](https://github.com/naiiytom/nist-frvt-archive/releases/download/v1.0/model.tar.gz) and [lib.tar.gz](https://github.com/naiiytom/nist-frvt-archive/releases/download/v1.0/lib.tar.gz) from [release](https://github.com/naiiytom/nist-frvt-archive/releases/) section and extract them accordingly.

.
## Update

**v1.0**
- initial commit
- update libraries, model and result

**v1.1**
- fixed mxnet using threading
- fixed image size from dynanic -> static size of 640x480 pixel

**v1.2**
- fixed stdout from mxnet 3rd party library

**v1.3**
- clear unused libraries
- minimize libraries size

**v1.rc-1**
- change retinaface backbone from ResNet50 -> MobileNet
- remove stdout from test code

---

## Download section

| Release | Model | Library |
|---------|-------|---------|
| [v1.0](https://github.com/naiiytom/nist-frvt-archive/releases/tag/v1.0) | [model.tar.gz](https://github.com/naiiytom/nist-frvt-archive/releases/download/v1.0/model.tar.gz) | [lib.tar.gz](https://github.com/naiiytom/nist-frvt-archive/releases/download/v1.0/lib.tar.gz) |
| [v1.1](https://github.com/naiiytom/nist-frvt-archive/releases/tag/v1.1) | [model.tar.gz](https://github.com/naiiytom/nist-frvt-archive/releases/download/v1.1/model.tar.gz) | [lib.tar.gz](https://github.com/naiiytom/nist-frvt-archive/releases/download/v1.1/lib.tar.gz) |
| [v1.2](https://github.com/naiiytom/nist-frvt-archive/releases/tag/v1.2) | [model.tar.gz](https://github.com/naiiytom/nist-frvt-archive/releases/download/v1.2/model.tar.gz) | [lib.tar.gz](https://github.com/naiiytom/nist-frvt-archive/releases/download/v1.2/lib.tar.gz) |
| [v1.3](https://github.com/naiiytom/nist-frvt-archive/releases/tag/v1.3) | [model.tar.gz](https://github.com/naiiytom/nist-frvt-archive/releases/download/v1.3/model.tar.gz) | [lib.tar.gz](https://github.com/naiiytom/nist-frvt-archive/releases/download/v1.3/lib.tar.gz) |
| [v1.rc-1](https://github.com/naiiytom/nist-frvt-archive/releases/tag/v1.rc-1) | [model.tar.gz](https://github.com/naiiytom/nist-frvt-archive/releases/download/v1.rc-1/model.tar.gz) | [lib.tar.gz](https://github.com/naiiytom/nist-frvt-archive/releases/download/v1.rc-1/lib.tar.gz) |

---

## Face Recognition Vendor Test (FRVT) Validation Packages
This repository contains validation packages for all [Ongoing FRVT evaluation](https://www.nist.gov/programs-projects/face-recognition-vendor-test-frvt-ongoing) tracks.
We recommend developers clone the entire repository and run validation from within
the folder that corresponds to the evaluation of interest.  The ./common directory
contains files that are shared across all validation packages.

