import pandas as pd
import json
from sklearn.model_selection import train_test_split
import os

def split_dataset():
    """
    Divide el dataset CSV en entrenamiento (80%) y validación (20%)
    y guarda los archivos JSON para el entrenamiento
    """
    
    print("🚀 Iniciando división del dataset...")
    
    # Verificar que el archivo existe
    if not os.path.exists('preguntas.csv'):
        print("❌ Error: No se encontró el archivo 'preguntas.csv'")
        print("   Asegúrate de que el archivo esté en el mismo directorio que este script")
        return False
    
    try:
        # Cargar el CSV
        print("📊 Cargando dataset desde preguntas.csv...")
        df = pd.read_csv('preguntas.csv', encoding='utf-8')
        
        print(f"   Dataset original: {len(df)} ejemplos")
        print(f"   Columnas encontradas: {list(df.columns)}")
        
        # Verificar que las columnas necesarias existen
        if 'question' not in df.columns or 'answer' not in df.columns:
            print("❌ Error: El CSV debe tener columnas 'question' y 'answer'")
            print(f"   Columnas encontradas: {list(df.columns)}")
            return False
        
        # Limpiar datos (eliminar filas con valores nulos)
        original_size = len(df)
        df = df.dropna(subset=['question', 'answer'])
        
        if len(df) < original_size:
            print(f"   Se eliminaron {original_size - len(df)} filas con datos faltantes")
        
        # Eliminar filas donde question o answer estén vacías
        df = df[(df['question'].str.strip() != '') & (df['answer'].str.strip() != '')]
        print(f"   Dataset limpio: {len(df)} ejemplos válidos")
        
        if len(df) < 10:
            print("❌ Error: Dataset muy pequeño después de la limpieza")
            return False
        
        # Dividir en train/validación (80/20)
        train_df, val_df = train_test_split(
            df, 
            test_size=0.2, 
            random_state=42,  # Semilla fija para reproducibilidad
            shuffle=True
        )
        
        print(f"\n📈 División completada:")
        print(f"   - Entrenamiento: {len(train_df)} ejemplos ({len(train_df)/len(df)*100:.1f}%)")
        print(f"   - Validación: {len(val_df)} ejemplos ({len(val_df)/len(df)*100:.1f}%)")
        
        # Crear estructuras JSON
        def create_json_structure(dataframe):
            """Convierte DataFrame a estructura JSON compatible"""
            data_list = []
            for _, row in dataframe.iterrows():
                data_list.append({
                    "question": str(row['question']).strip(),
                    "answer": str(row['answer']).strip()
                })
            return {"data": data_list}
        
        # Generar estructuras JSON
        train_json = create_json_structure(train_df)
        val_json = create_json_structure(val_df)
        
        # Guardar archivos JSON
        print("\n💾 Guardando archivos...")
        
        # Guardar train_dataset.json
        with open('train_dataset.json', 'w', encoding='utf-8') as f:
            json.dump(train_json, f, ensure_ascii=False, indent=2)
        print(f"   ✅ train_dataset.json guardado ({len(train_json['data'])} ejemplos)")
        
        # Guardar validation_dataset.json
        with open('validation_dataset.json', 'w', encoding='utf-8') as f:
            json.dump(val_json, f, ensure_ascii=False, indent=2)
        print(f"   ✅ validation_dataset.json guardado ({len(val_json['data'])} ejemplos)")
        
        # También guardar como CSV por si acaso
        train_df.to_csv('train_dataset.csv', index=False, encoding='utf-8')
        val_df.to_csv('validation_dataset.csv', index=False, encoding='utf-8')
        print(f"   ✅ Archivos CSV de respaldo también guardados")
        
        # Mostrar ejemplos
        print(f"\n🔍 Ejemplo del dataset de entrenamiento:")
        example = train_json['data'][0]
        print(f"   Pregunta: {example['question'][:80]}...")
        print(f"   Respuesta: {example['answer'][:80]}...")
        
        print(f"\n🔍 Ejemplo del dataset de validación:")
        example = val_json['data'][0]
        print(f"   Pregunta: {example['question'][:80]}...")
        print(f"   Respuesta: {example['answer'][:80]}...")
        
        # Estadísticas adicionales
        print(f"\n📊 Estadísticas:")
        avg_q_len = train_df['question'].str.len().mean()
        avg_a_len = train_df['answer'].str.len().mean()
        print(f"   Longitud promedio de preguntas: {avg_q_len:.0f} caracteres")
        print(f"   Longitud promedio de respuestas: {avg_a_len:.0f} caracteres")
        
        # Verificar que los archivos se guardaron correctamente
        if os.path.exists('train_dataset.json') and os.path.exists('validation_dataset.json'):
            train_size = os.path.getsize('train_dataset.json')
            val_size = os.path.getsize('validation_dataset.json')
            print(f"\n📁 Archivos creados exitosamente:")
            print(f"   - train_dataset.json: {train_size/1024:.1f} KB")
            print(f"   - validation_dataset.json: {val_size/1024:.1f} KB")
        
        print(f"\n🎯 ¡División completada exitosamente!")
        print(f"   Los archivos están listos para usar con tu código de entrenamiento")
        print(f"   Usa 'train_dataset.json' y 'validation_dataset.json' en tu configuración")
        
        return True
        
    except FileNotFoundError:
        print("❌ Error: No se pudo encontrar el archivo 'preguntas.csv'")
        return False
    except pd.errors.EmptyDataError:
        print("❌ Error: El archivo CSV está vacío")
        return False
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        return False

def verify_files():
    """Verifica que los archivos JSON se crearon correctamente"""
    print("\n🔍 Verificando archivos creados...")
    
    files_to_check = ['train_dataset.json', 'validation_dataset.json']
    
    for filename in files_to_check:
        if os.path.exists(filename):
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                if 'data' in data and len(data['data']) > 0:
                    print(f"   ✅ {filename}: {len(data['data'])} ejemplos - OK")
                    
                    # Verificar estructura del primer ejemplo
                    first_example = data['data'][0]
                    if 'question' in first_example and 'answer' in first_example:
                        print(f"      Estructura correcta: question y answer presentes")
                    else:
                        print(f"      ⚠️ Advertencia: Estructura incorrecta en {filename}")
                else:
                    print(f"   ❌ {filename}: Estructura de datos incorrecta")
            except json.JSONDecodeError:
                print(f"   ❌ {filename}: Error al leer JSON")
            except Exception as e:
                print(f"   ❌ {filename}: Error - {e}")
        else:
            print(f"   ❌ {filename}: Archivo no encontrado")

if __name__ == "__main__":
    # Ejecutar la división
    success = split_dataset()
    
    if success:
        # Verificar los archivos creados
        verify_files()
        
        print(f"\n" + "="*60)
        print(f"🎉 PROCESO COMPLETADO EXITOSAMENTE")
        print(f"="*60)
        print(f"📁 Archivos generados:")
        print(f"   • train_dataset.json (para entrenamiento)")
        print(f"   • validation_dataset.json (para validación)")
        print(f"   • train_dataset.csv (respaldo)")
        print(f"   • validation_dataset.csv (respaldo)")
        print(f"\n🚀 Ahora puedes usar el código de entrenamiento actualizado")
        print(f"   que incluye Early Stopping y validación automática")
    else:
        print(f"\n❌ No se pudo completar la división del dataset")
        print(f"   Revisa los errores mostrados arriba")