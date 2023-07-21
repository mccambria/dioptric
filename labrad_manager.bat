:: This file may have to be tweaked for your specific PC. Just consider it a starting point. 
start cmd /k %UserProfile%\Documents\Github\dioptric\manager\scalabrad-0.8.3\bin\labrad.bat
start cmd /k %UserProfile%\Documents\Github\dioptric\manager\scalabrad-web-server-2.0.5\bin\labrad-web.bat
start chrome /new-window http://localhost:7667
call %UserProfile%\miniconda3\Scripts\activate dioptric
call python -m labrad.node