
REM for /l %%i in (1,1,30) do python keras_fine_tune_weights_name.py --model_dir C:\tmp\InceptionResNet\skirt_length_labels
REM for /l %%i in (1,1,30) do python keras_fine_tune_weights_name.py --model_dir C:\tmp\InceptionResNet\sleeve_length_labels
for /l %%i in (1,1,30) do python keras_fine_tune_weights_name.py --model_dir C:\tmp\InceptionResNet\pant_length_labels
for /l %%i in (1,1,30) do python keras_fine_tune_weights_name.py --model_dir C:\tmp\InceptionResNet\neckline_design_labels
for /l %%i in (1,1,30) do python keras_fine_tune_weights_name.py --model_dir C:\tmp\InceptionResNet\neck_design_labels
for /l %%i in (1,1,30) do python keras_fine_tune_weights_name.py --model_dir C:\tmp\InceptionResNet\lapel_design_labels
for /l %%i in (1,1,30) do python keras_fine_tune_weights_name.py --model_dir C:\tmp\InceptionResNet\collar_design_labels
for /l %%i in (1,1,30) do python keras_fine_tune_weights_name.py --model_dir C:\tmp\InceptionResNet\coat_length_labels
pause