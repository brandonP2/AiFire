"""
project_config.py - Configuración dinámica del proyecto

Este módulo maneja la carga de config.yaml y calcula automáticamente
el project_folder, haciendo la configuración portable entre ambientes.

Usage:
    from project_config import load_config, get_project_root

    config = load_config()
    print(config['project_folder'])  # Se calcula automáticamente
"""

import yaml
from pathlib import Path
import os
import sys


def load_env_file(env_path=None):
    """
    Carga variables de entorno desde un archivo .env

    Args:
        env_path: Ruta al archivo .env. Si es None, busca automáticamente:
                  1. Variable PF_PYTHON_ENV (ruta absoluta)
                  2. .env.development en la RAÍZ del proyecto
                  3. .env en la RAÍZ del proyecto

    Funciona tanto en local como en Docker sin fricción.
    """
    if env_path is None:
        # Prioridad 1: Variable de entorno PF_PYTHON_ENV (Docker o export manual)
        env_path = os.getenv('PF_PYTHON_ENV')

        # Si no hay PF_PYTHON_ENV, buscar en la raíz del proyecto
        if not env_path:
            # Detectar raíz del proyecto
            try:
                project_root = get_project_root()
            except:
                # Si falla, usar directorio actual
                project_root = Path.cwd()

            # Prioridad 2: .env.development en raíz del proyecto
            env_dev = project_root / '.env.development'
            if env_dev.exists():
                env_path = str(env_dev)

            # Prioridad 3: .env en raíz del proyecto
            if not env_path:
                env_file = project_root / '.env'
                if env_file.exists():
                    env_path = str(env_file)

    if not env_path:
        return  # No hay archivo .env, continuar sin error

    env_path = Path(env_path)
    if not env_path.exists():
        return  # Archivo no existe, continuar sin error

    # Cargar variables
    loaded_count = 0
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            # Ignorar comentarios y líneas vacías
            if not line or line.startswith('#'):
                continue

            # Separar key=value
            if '=' in line:
                key, value = line.split('=', 1)
                # Quitar comillas si existen
                value = value.strip().strip('"').strip("'")
                os.environ[key.strip()] = value
                loaded_count += 1

    # Debug info (solo si se cargaron variables)
    if loaded_count > 0:
        print(f"✅ Variables de entorno cargadas desde: {env_path.name} ({loaded_count} variables)")


def get_project_root() -> Path:
    """
    Detecta automáticamente la raíz del proyecto.

    Busca hacia arriba desde el directorio actual hasta encontrar:
    - pyproject.toml
    - src/config/config.yaml
    - README.md

    Returns:
        Path: Ruta absoluta a la raíz del proyecto
    """
    # Empezar desde el directorio del archivo actual
    current = Path(__file__).resolve().parent

    # Buscar hacia arriba hasta encontrar marcadores del proyecto
    for parent in [current] + list(current.parents):
        # Buscar archivos que indiquen la raíz del proyecto
        markers = [
            parent / 'pyproject.toml',
            parent / 'src' / 'config' / 'config.yaml',
            parent / '.python-version',
        ]

        if any(marker.exists() for marker in markers):
            return parent

    # Si no encuentra, usar el directorio actual
    return Path.cwd()


def get_config_path(project_root: Path = None) -> Path:
    """
    Obtiene la ruta al archivo config.yaml.

    Args:
        project_root: Raíz del proyecto (se detecta automáticamente si no se provee)

    Returns:
        Path: Ruta al config.yaml
    """
    if project_root is None:
        project_root = get_project_root()

    return project_root / 'src' / 'config' / 'config.yaml'


def _substitute_env_vars(obj):
    """
    Reemplaza ${VAR_NAME} con valores de variables de entorno.

    Convierte automáticamente tipos:
    - "5432" → 5432 (int)
    - "true" → True (bool)
    - "false" → False (bool)

    Args:
        obj: Diccionario, lista o valor a procesar

    Returns:
        Objeto con variables sustituidas y tipos correctos
    """
    if isinstance(obj, dict):
        return {k: _substitute_env_vars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_substitute_env_vars(item) for item in obj]
    elif isinstance(obj, str):
        # Buscar patrón ${VAR_NAME}
        import re
        pattern = r'\$\{([^}]+)\}'
        matches = re.findall(pattern, obj)

        result = obj
        for var_name in matches:
            var_value = os.getenv(var_name)
            if var_value is not None:
                result = result.replace(f'${{{var_name}}}', var_value)

        # Si el string original era SOLO una variable (ej: "${port}"),
        # intentar convertir a tipo apropiado
        if obj.startswith('${') and obj.endswith('}'):
            # Era solo una variable, intentar convertir tipo
            return _auto_convert_type(result)

        return result
    else:
        return obj


def _auto_convert_type(value: str):
    """
    Convierte string a tipo apropiado automáticamente.

    Args:
        value: String a convertir

    Returns:
        Valor con tipo apropiado (int, bool, float, o str)
    """
    if not isinstance(value, str):
        return value

    # Intentar int
    try:
        return int(value)
    except ValueError:
        pass

    # Intentar float
    try:
        return float(value)
    except ValueError:
        pass

    # Intentar bool
    if value.lower() in ('true', 'yes', '1'):
        return True
    if value.lower() in ('false', 'no', '0'):
        return False

    # Retornar como string
    return value


def load_config(config_path: str = None) -> dict:
    """
    Carga config.yaml y calcula project_folder automáticamente.

    Automáticamente:
    1. Carga variables de entorno desde .env file
    2. Sustituye ${VAR_NAME} en el config con valores reales

    Args:
        config_path: Ruta al config.yaml (opcional, se detecta automáticamente)

    Returns:
        dict: Configuración con project_folder calculado automáticamente

    Example:
        >>> config = load_config()
        >>> print(config['project_folder'])
        '/home/user/project'
        >>> print(config['data']['raw']['local_path'])
        'data/raw'
    """
    # 1. Cargar variables de entorno desde .env
    load_env_file()

    # 2. Detectar raíz del proyecto
    project_root = get_project_root()

    # 3. Determinar ruta del config
    if config_path is None:
        config_path = get_config_path(project_root)
    else:
        config_path = Path(config_path)

    # 4. Cargar config.yaml
    if not config_path.exists():
        raise FileNotFoundError(
            f"No se encontró config.yaml en: {config_path}\n"
            f"Raíz del proyecto detectada: {project_root}"
        )

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 5. Sustituir variables de entorno ${VAR_NAME}
    config = _substitute_env_vars(config)

    # 6. Sobrescribir project_folder con el valor calculado
    config['project_folder'] = str(project_root)

    # 7. Agregar path helpers
    config['_paths'] = {
        'root': project_root,
        'src': project_root / 'src',
        'data': project_root / 'data',
        'notebooks': project_root / 'notebooks',
        'queries': project_root / 'src' / 'queries',
        'config': project_root / 'src' / 'config',
    }

    return config


def setup_imports(config: dict = None):
    """
    Agrega src/ al sys.path para permitir imports.

    Llama a esta función al inicio de cada notebook/script.

    Args:
        config: Configuración (se carga automáticamente si no se provee)

    Example:
        >>> from project_config import setup_imports
        >>> setup_imports()
        >>> import utils  # Ahora funciona
        >>> import local_storage  # Ahora funciona
    """
    if config is None:
        config = load_config()

    src_path = str(Path(config['project_folder']) / 'src')

    if src_path not in sys.path:
        sys.path.insert(1, src_path)
        print(f"✅ Added to sys.path: {src_path}")


def get_absolute_path(relative_path: str, config: dict = None) -> Path:
    """
    Convierte una ruta relativa en absoluta usando project_folder.

    Args:
        relative_path: Ruta relativa (ej: 'data/raw/features.parquet')
        config: Configuración (se carga automáticamente si no se provee)

    Returns:
        Path: Ruta absoluta

    Example:
        >>> path = get_absolute_path('data/raw/features.parquet')
        >>> print(path)
        /home/user/project/data/raw/features.parquet
    """
    if config is None:
        config = load_config()

    project_root = Path(config['project_folder'])
    return project_root / relative_path


# =============================================================================
# Uso en notebooks - Patrón recomendado
# =============================================================================

def init_notebook():
    """
    Inicializa un notebook con configuración y imports.

    Llama a esto al inicio de cada notebook para:
    1. Cargar config.yaml automáticamente
    2. Configurar sys.path para imports
    3. Retornar configuración lista para usar

    Returns:
        dict: Configuración del proyecto

    Example en notebook:
        >>> from project_config import init_notebook
        >>> config = init_notebook()
        >>> # Ahora puedes usar:
        >>> import utils
        >>> import local_storage
        >>> print(config['project_folder'])
    """
    config = load_config()
    setup_imports(config)

    print(f"📂 Project: {config['project_name']}")
    print(f"📁 Root: {config['project_folder']}")
    print(f"✅ Config loaded & imports configured")

    return config


# =============================================================================
# Para compatibilidad con código legacy
# =============================================================================

def load_config_legacy(config_path: str) -> dict:
    """
    Versión legacy para compatibilidad con código antiguo.

    Esta función se comporta igual que antes, pero agrega
    project_folder calculado automáticamente si no está presente.

    Args:
        config_path: Ruta al config.yaml

    Returns:
        dict: Configuración
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Si no tiene project_folder o es placeholder, calcularlo
    if 'project_folder' not in config or config['project_folder'].startswith('/path/to'):
        config['project_folder'] = str(get_project_root())

    return config


# =============================================================================
# Ejemplo de uso
# =============================================================================

if __name__ == "__main__":
    # Test
    print("=" * 60)
    print("Testing project_config.py")
    print("=" * 60)

    # 1. Detectar raíz del proyecto
    root = get_project_root()
    print(f"\n1. Project root: {root}")

    # 2. Cargar config
    config = load_config()
    print(f"\n2. Config loaded:")
    print(f"   - project_name: {config['project_name']}")
    print(f"   - project_folder: {config['project_folder']}")

    # 3. Configurar imports
    setup_imports(config)

    # 4. Paths helpers
    print(f"\n3. Path helpers:")
    print(f"   - src: {config['_paths']['src']}")
    print(f"   - data: {config['_paths']['data']}")
    print(f"   - queries: {config['_paths']['queries']}")

    # 5. Absolute path
    abs_path = get_absolute_path('data/raw/features.parquet', config)
    print(f"\n4. Absolute path example:")
    print(f"   'data/raw/features.parquet' → {abs_path}")

    print("\n✅ All tests passed!")
