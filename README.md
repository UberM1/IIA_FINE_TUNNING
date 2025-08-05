# sandIA

## Carpetas

### book_trained

Aqui se encuentran todos los archivos que se usaron para entrenar al modelo con el libro. 
En data.txt se encuentra el libro completo y es el archivo del cual el modelo recupera la informacion 
de la que aprende. En data_old.txt están los primeros tres capítulos del libro. out.txt es el archiivo 
generado por el script que transforma el libro de EPUB a txt. El libro original se encuentra en 
The Way of Kings.epub. Los archivos parser.py y remove_images.lua fueron usados para realizar este paso 
del libro a txt. 

El modelo base se toma desde huggingface y se entrena con tp.py, allí se encuentra toda la configuración 
para esto. Una vez entrenado, se generan otras tres carpetas, llama-2-7b-chat-hf-sandia es en la que se 
guarda luego del entrenamiento; results guarda checkpoints y otra metadata relacionada al entrenamiento; 
sandia_merged contiene también datos internos relacionados al modelo entrenado. Solo entregamos la primera 
de las tres, ya que las otras dos son demasiado pesadas. Si se quiere entrenar el modelo puede hacerse con 
tp.py y se generarán las carpetas correspondientes (se recomienda modificar la configuración y usar el archivo 
data_old.txt). También se encuentra el archivo requirements.txt, que está presente sólo para evitar algunos 
posibles errores por el uso de huggingface. 

Las preguntas que se quieran hacer, y respuestas dadas por el modelo se encuentran en preguntas.txt, cuando 
se corre query.py luego de entrenar, el modelo toma las preguntas del principio de preguntas.txt y las respuestas 
se concatenan al final de este. Este archivo es muy extenso, pero contiene toda la información de cómo fueron 
las configuraciones de entrenamiento usadas y qué respuestas entregó el modelo en cada caso.

También adjuntamos el archivo Preguntas y respuestas.txt que usamos para documentar fases muy tempranas de 
entrenamiento antes de pasar a usar el archivo preguntas.txt.

### ia_powered

En esta carpeta están todos los archivos relacionados al entrenamiento con los pares de preguntas y respuestas 
generados con Claude. Los archivos específicos para esto son IA_POWERED.json, IA_POWERED.txt, split.py, 
train_dataset.csv, train_dataset.json, validation_dataset.csv y validation_dataset.json. Donde split.py genera 
las divisiones correspondientes de los datasets.

Para este entrenamiento el modelo sigue tomándose de huggingface, pero se entrena con train.py. Como en el caso 
anterior, cuando se entrena se generan también tres carpetas como se dijo anteriormente. Nuevamente, solo entregamos 
llama-2-7b-chat-hf-sandia.

Las preguntas también se encuentran en preguntas.txt y las respuestas se concatenan al final de este archivo cuando 
se corre query.py luego de entrenar.
