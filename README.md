# MRFScore XGBoost Regression Model
The MRF Score, standing for Music Relationship Feature, represents the music relationship score of a song or track.
Originally, extracting MRF Scores requires the pitch notes of a song or track.
However, to simplify this process, we developed a regression model using the XGBoost Regressor.
Given that the XGBoost Regressor demands numerous parameters, we employed an evolutionary algorithm to optimize these parameters.


## Library Requirements
- anyio==3.6.2
- appdirs==1.4.4
- astunparse==1.6.3
- asynctest==0.13.0
- audioread==3.0.0
- blinker==1.4
- brotlipy==0.7.0
- cachetools==5.2.0
- charset-normalizer==2.1.1
- decorator==5.1.1
- ffmpeg-python==0.2.0
- flatbuffers==22.12.6
- future==0.18.2
- google-auth==2.15.0
- google-auth-oauthlib==0.4.6
- grpcio==1.51.1
- h11==0.12.0
-  h2==4.1.0
- hpack==4.0.0
- httpcore==0.13.7
- httpx==0.19.0
- hyperframe==6.0.1
- joblib==1.2.0
- libclang==14.0.6
- librosa==0.8.1
- llvmlite==0.38.1
- mkl-fft==1.3.1
- mkl-service==2.4.0
- norbert==0.2.1
- numba==0.55.2
- numpy==1.21.6
- oauthlib==3.2.2
- pandas==1.3.5
- pooch==1.6.0
- protobuf==3.19.6
- pyasn1-modules==0.2.8
- python-dateutil==2.8.2
- pytz==2022.7
- requests-oauthlib==1.3.1
- resampy==0.4.2
- rfc3986==1.5.0
- rsa==4.9
- scikit-learn==1.0.2
- sniffio==1.3.0
- soundfile==0.11.0
- spleeter==2.3.2
- tensorboard==2.10.1
- tensorflow-io-gcs-filesystem==0.29.0
- termcolor==2.1.1
- threadpoolctl==3.1.0
- typer==0.3.2
- zipp==3.11.0
- xgboost==1.6.2




### License
Duksung License

Copyright (c) 2024 AIoT Lab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
