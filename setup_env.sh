#!/bin/bash

# Script para crear un entorno de miniconda con las librerías del requirements.txt
# Autor: Script generado automáticamente
# Fecha: $(date)

set -e  # Salir si cualquier comando falla

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Función para imprimir mensajes con color
print_message() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================${NC}"
}

# Verificar si miniconda está instalado
check_miniconda() {
    if command -v conda &> /dev/null; then
        print_message "Miniconda encontrado en: $(which conda)"
        print_message "Versión de conda: $(conda --version)"
        return 0
    else
        print_error "Miniconda no está instalado o no está en el PATH"
        print_message "Por favor instala miniconda desde: https://docs.conda.io/en/latest/miniconda.html"
        return 1
    fi
}

# Verificar si existe el archivo requirements.txt
check_requirements() {
    if [ ! -f "requirements.txt" ]; then
        print_error "No se encontró el archivo requirements.txt en el directorio actual"
        print_message "Asegúrate de ejecutar este script desde el directorio del proyecto"
        return 1
    fi
    print_message "Archivo requirements.txt encontrado"
    return 0
}

# Crear el entorno conda
create_conda_env() {
    local env_name="mlops_lab03"
    
    print_header "CREANDO ENTORNO CONDA"
    
    # Verificar si el entorno ya existe
    if conda env list | grep -q "^${env_name}\s"; then
        print_warning "El entorno '${env_name}' ya existe"
        read -p "¿Deseas eliminar el entorno existente y crear uno nuevo? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_message "Eliminando entorno existente..."
            conda env remove -n "${env_name}" -y
        else
            print_message "Usando entorno existente"
            return 0
        fi
    fi
    
    # Crear nuevo entorno con Python 3.9
    print_message "Creando nuevo entorno '${env_name}' con Python 3.9..."
    conda create -n "${env_name}" python=3.9 -y
    
    print_message "Entorno creado exitosamente"
}

# Instalar dependencias
install_dependencies() {
    local env_name="mlops_lab03"
    
    print_header "INSTALANDO DEPENDENCIAS"
    
    print_message "Activando entorno..."
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${env_name}"
    
    # Actualizar pip
    print_message "Actualizando pip..."
    pip install --upgrade pip
    
    # Instalar dependencias desde requirements.txt
    print_message "Instalando dependencias desde requirements.txt..."
    pip install -r requirements.txt
    
    print_message "Todas las dependencias instaladas exitosamente"
}

# Mostrar información del entorno
show_env_info() {
    local env_name="mlops_lab03"
    
    print_header "INFORMACIÓN DEL ENTORNO"
    
    print_message "Nombre del entorno: ${env_name}"
    print_message "Para activar el entorno, ejecuta:"
    echo -e "${YELLOW}conda activate ${env_name}${NC}"
    print_message "Para desactivar el entorno, ejecuta:"
    echo -e "${YELLOW}conda deactivate${NC}"
    
    # Mostrar paquetes instalados
    print_message "Paquetes instalados en el entorno:"
    conda list
}

# Función principal
main() {
    print_header "CONFIGURACIÓN DE ENTORNO MLOPS LAB03"
    
    # Verificaciones previas
    if ! check_miniconda; then
        exit 1
    fi
    
    if ! check_requirements; then
        exit 1
    fi
    
    # Crear entorno e instalar dependencias
    create_conda_env
    install_dependencies
    
    print_header "CONFIGURACIÓN COMPLETADA"
    print_message "El entorno ha sido configurado exitosamente"
    
    show_env_info
    
    print_message "¡Listo para trabajar! 🚀"
}

# Ejecutar función principal
main "$@"
