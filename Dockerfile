# Usa la imagen base oficial de Python
FROM python:3.9-slim

# Establece el directorio de trabajo en /app
WORKDIR /app

# Copia los archivos necesarios (asegúrate de tener requirements.txt)
COPY requirements.txt .

# Instala las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copia el resto de los archivos de la aplicación
COPY . .

# Expone el puerto 5000 (o el puerto que uses para Flask)
EXPOSE 5000

# Comando para ejecutar la aplicación Flask en el servidor WSGI Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]
