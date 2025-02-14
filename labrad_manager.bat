:: This file may have to be tweaked for your specific PC. Just consider it a starting point. 
start cmd /k ""C:\Users\Saroj Chand\Documents\dioptric\manager\scalabrad-0.8.3\bin\labrad.bat""
start cmd /k ""C:\Users\Saroj Chand\Documents\dioptric\manager\scalabrad-web-server-2.0.5\bin\labrad-web.bat""
start chrome /new-window http://localhost:7667
call "C:\Users\Saroj Chand\miniconda3\Scripts\activate.bat" dioptric
call python -m labrad.node -u "" -w ""
