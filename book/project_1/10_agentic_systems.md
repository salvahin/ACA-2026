---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Lectura 10: Sistemas Agénticos

```{admonition} Ejecutar en Google Colab
:class: tip

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/salvahin/ACA-2026/blob/main/book/notebooks/10_agentic_systems.ipynb)
```

```{code-cell} ipython3
:tags: [remove-input, setup]

# Setup Colab Environment
!pip install -q plotly
print('Dependencies installed!')
```

```{admonition} Objetivos de Aprendizaje
:class: tip
Al finalizar esta lectura podrás:
- Implementar tool use / function calling para extender capacidades de LLMs
- Aplicar el patrón ReAct (Reasoning + Acting) para razonamiento explícito
- Diseñar sistemas RAG (Retrieval Augmented Generation) con vector stores
- Arquitectar sistemas multi-agente con especialización y colaboración
- Evaluar tendencias futuras: razonamiento verificable, auto-improvement, compositionality
```

```{admonition} Milestone del Proyecto
:class: important
Después de esta lectura podrás: Transformar KernelAgent de un generador simple a un sistema agéntico completo. Implementarás herramientas (compilar Triton, ejecutar benchmarks), RAG con documentación de Triton, y un agente revisor que mejora kernels iterativamente. El sistema resultante es autónomo, transparente y mejorable.
```

## Introducción

Un LLM es poderoso, pero limitado: genera texto. Un **sistema agéntico** es una arquitectura donde un LLM actúa como "cerebro", decidiendo qué herramientas usar, cómo planificar, y cómo razonar sobre problemas complejos.

Esta lectura final explora tool use, multi-agent architectures, RAG, reasoning, y el futuro de IA estructurada.

---

## Parte 1: Tool Use / Function Calling

### El Problema Básico

```
Pregunta: "¿Cuál es el precio de Bitcoin hoy?"

LLM (sin herramientas):
  "Bitcoin fluctúa constantemente. Mi conocimiento es hasta Abril 2024..."
  ← No puede responder con información actual

LLM (con herramientas):
  1. Recognizes pregunta requiere información actual
  2. Calls tool: get_bitcoin_price()
  3. Obtiene: $62,000
  4. Responde: "Bitcoin está a $62,000 hoy"
```

### Function Calling Schema

Especificas qué funciones puede usar el LLM:

```{code-cell} ipython3
# Sistema de Function Calling para LLMs
import json
from typing import Dict, List, Any, Callable
from datetime import datetime

class FunctionRegistry:
    """Registro de funciones disponibles para el LLM"""

    def __init__(self):
        self.functions: Dict[str, Dict] = {}
        self.implementations: Dict[str, Callable] = {}

    def register(self, name: str, description: str, parameters: Dict,
                implementation: Callable):
        """Registra una función disponible para el LLM"""
        schema = {
            "name": name,
            "description": description,
            "parameters": parameters
        }
        self.functions[name] = schema
        self.implementations[name] = implementation

    def get_schema(self) -> List[Dict]:
        """Retorna el schema de todas las funciones"""
        return list(self.functions.values())

    def execute(self, function_name: str, arguments: Dict) -> Any:
        """Ejecuta una función registrada"""
        if function_name not in self.implementations:
            raise ValueError(f"Función no encontrada: {function_name}")

        return self.implementations[function_name](**arguments)

    def format_for_llm(self) -> str:
        """Formatea las funciones para el prompt del LLM"""
        formatted = "Available functions:\n\n"
        for func in self.functions.values():
            formatted += f"- {func['name']}: {func['description']}\n"
            formatted += f"  Parameters: {json.dumps(func['parameters'], indent=2)}\n\n"
        return formatted

# Crear registro de funciones
registry = FunctionRegistry()

# Función 1: Obtener precio de Bitcoin (simulado)
def get_bitcoin_price() -> Dict:
    """Simula obtener precio de Bitcoin"""
    import random
    price = 60000 + random.randint(-5000, 5000)
    return {
        "price": price,
        "currency": "USD",
        "timestamp": datetime.now().isoformat()
    }

registry.register(
    name="get_bitcoin_price",
    description="Obtiene el precio actual de Bitcoin en USD",
    parameters={
        "type": "object",
        "properties": {},  # Sin parámetros
        "required": []
    },
    implementation=get_bitcoin_price
)

# Función 2: Buscar en web (simulado)
def search_web(query: str) -> List[Dict]:
    """Simula búsqueda web"""
    return [
        {
            "title": f"Resultado 1 para '{query}'",
            "snippet": "Información relevante sobre la búsqueda...",
            "url": "https://example.com/1"
        },
        {
            "title": f"Resultado 2 para '{query}'",
            "snippet": "Más información sobre el tema...",
            "url": "https://example.com/2"
        }
    ]

registry.register(
    name="search_web",
    description="Busca información en internet",
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "La consulta de búsqueda"
            }
        },
        "required": ["query"]
    },
    implementation=search_web
)

# Función 3: Enviar email (simulado)
def send_email(to: str, subject: str, body: str) -> Dict:
    """Simula envío de email"""
    print(f"\n[EMAIL ENVIADO]")
    print(f"To: {to}")
    print(f"Subject: {subject}")
    print(f"Body: {body[:100]}...")
    return {
        "status": "sent",
        "message_id": "msg_12345",
        "timestamp": datetime.now().isoformat()
    }

registry.register(
    name="send_email",
    description="Envía un email",
    parameters={
        "type": "object",
        "properties": {
            "to": {
                "type": "string",
                "description": "Dirección de email del destinatario"
            },
            "subject": {
                "type": "string",
                "description": "Asunto del email"
            },
            "body": {
                "type": "string",
                "description": "Cuerpo del email"
            }
        },
        "required": ["to", "subject", "body"]
    },
    implementation=send_email
)

# Mostrar funciones disponibles
print("FUNCIONES REGISTRADAS:")
print("="*60)
print(registry.format_for_llm())

# Ejemplo de uso
print("\nEJEMPLOS DE EJECUCIÓN:")
print("="*60)

# Ejecutar get_bitcoin_price
print("\n1. Obtener precio de Bitcoin:")
result = registry.execute("get_bitcoin_price", {})
print(f"   Precio: ${result['price']:,} {result['currency']}")
print(f"   Timestamp: {result['timestamp']}")

# Ejecutar search_web
print("\n2. Buscar en web:")
results = registry.execute("search_web", {"query": "machine learning"})
for i, result in enumerate(results, 1):
    print(f"   Resultado {i}: {result['title']}")

# Ejecutar send_email
print("\n3. Enviar email:")
result = registry.execute(
    "send_email",
    {
        "to": "usuario@example.com",
        "subject": "Resumen de investigación",
        "body": "Aquí está el resumen de la investigación sobre IA..."
    }
)
print(f"   Status: {result['status']}")
print(f"   Message ID: {result['message_id']}")

# Schema JSON para el LLM
print("\n" + "="*60)
print("SCHEMA JSON COMPLETO:")
print(json.dumps(registry.get_schema(), indent=2))
```

### Flujo de Ejecución

```
1. Usuario pregunta:
   "Investiga empresas de AI y envía resumen por email"

2. LLM entiende:
   - Necesita search_web para investigar
   - Necesita send_email para enviar resultado

3. Llama tool:
   search_web("empresas AI 2024")
   → Retorna: "OpenAI, Anthropic, Google DeepMind, ..."

4. Llama segunda tool:
   send_email(
     to="usuario@mail.com",
     subject="Resumen empresas AI",
     body="Las principales empresas AI en 2024 son..."
   )

5. Retorna a usuario:
   "Envié el resumen a tu email"
```

### Ventajas

```
✓ Información actual (browsing real-time)
✓ Acciones en sistemas externos (enviar email, actualizar BD)
✓ Cálculos precisos (en lugar de estimar)
✓ Separación clara de responsabilidades
```

---

## Parte 2: ReAct - Reasoning and Acting

**ReAct** = **RE**asoning + **ACT**ing. El LLM no solo actúa, sino que explica su razonamiento:

### Estructura ReAct

```
Thought → Action → Observation → Thought → ...

Ejemplo:
User: "¿Cuánto es 2847 * 1923?"

LLM:
Thought: Necesito calcular 2847 * 1923. Usaré la herramienta de cálculo.
Action: calculator(2847, 1923)
Observation: 5,466,381

Thought: Tengo el resultado, pero déjame verificar si es razonable.
         2847 ≈ 3000, 1923 ≈ 2000, entonces ~6,000,000. Mi respuesta de 5.4M es razonable.
         Puedo responder.

Answer: 2847 * 1923 = 5,466,381
```

### Por Qué Funciona

```
Sin ReAct:
  LLM intenta estimar mentalmente: "2847 * 1923 ≈ 5.5M"
  Problema: La aritmética es débil para LLMs

Con ReAct:
  LLM reconoce que necesita cálculo exacto
  Delega a herramienta especializada
  Verifica razonabilidad del resultado
  Responde con confianza
```

### Implementación

```{code-cell} ipython3
# Implementación de ReAct (Reasoning + Acting)
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import random

@dataclass
class ReActStep:
    """Un paso en el loop de ReAct"""
    thought: str
    action: Optional[str] = None
    observation: Optional[str] = None
    answer: Optional[str] = None

class MockLLM:
    """Mock de LLM para demostración de ReAct"""

    def __init__(self, tools):
        self.tools = tools

    def generate_react_step(self, history: List[ReActStep], query: str) -> ReActStep:
        """Simula generación de un paso ReAct"""
        # Para demostración, seguimos un script predefinido
        step_num = len(history)

        if step_num == 0:
            # Primer paso: identificar qué hacer
            return ReActStep(
                thought="Necesito calcular 2847 * 1923. Usaré la herramienta de cálculo.",
                action="calculator",
            )
        elif step_num == 1:
            # Segundo paso: verificar resultado
            return ReActStep(
                thought="Tengo el resultado. Déjame verificar si es razonable. "
                       "2847 ≈ 3000, 1923 ≈ 2000, entonces ~6,000,000. "
                       "Mi respuesta de 5.4M es razonable.",
                action="verify",
            )
        else:
            # Paso final: responder
            return ReActStep(
                thought="El cálculo es correcto y verificado.",
                answer="2847 * 1923 = 5,466,381"
            )

class ReActAgent:
    """Agente que implementa el patrón ReAct"""

    def __init__(self, tools: Dict[str, callable]):
        self.tools = tools
        self.llm = MockLLM(tools)
        self.max_iterations = 10

    def execute_tool(self, tool_name: str, *args, **kwargs) -> str:
        """Ejecuta una herramienta"""
        if tool_name not in self.tools:
            return f"Error: herramienta '{tool_name}' no encontrada"

        try:
            result = self.tools[tool_name](*args, **kwargs)
            return str(result)
        except Exception as e:
            return f"Error ejecutando {tool_name}: {str(e)}"

    def run(self, query: str) -> str:
        """Ejecuta el loop ReAct"""
        history = []
        print(f"User Query: {query}")
        print("="*70)

        for iteration in range(self.max_iterations):
            # Generar siguiente paso
            step = self.llm.generate_react_step(history, query)

            # Mostrar pensamiento
            print(f"\nIteration {iteration + 1}:")
            print(f"Thought: {step.thought}")

            # Si hay una acción, ejecutarla
            if step.action:
                print(f"Action: {step.action}")

                # Ejecutar herramienta
                if step.action == "calculator":
                    observation = self.execute_tool("calculator", 2847, 1923)
                elif step.action == "verify":
                    observation = "Verificación: 3000 * 2000 = 6,000,000. " \
                                "Resultado 5,466,381 está en el rango esperado."
                else:
                    observation = f"Ejecutando {step.action}..."

                step.observation = observation
                print(f"Observation: {observation}")

            # Si hay una respuesta, terminar
            if step.answer:
                print(f"\nFinal Answer: {step.answer}")
                return step.answer

            history.append(step)

        return "Error: se alcanzó el máximo de iteraciones"

# Herramientas disponibles
tools = {
    "calculator": lambda x, y: x * y,
    "search": lambda q: f"Resultados de búsqueda para: {q}",
    "weather": lambda city: f"El clima en {city} es soleado, 22°C"
}

# Crear agente
agent = ReActAgent(tools)

# Ejemplo 1: Cálculo matemático
print("\nEJEMPLO 1: Cálculo Matemático")
print("="*70)
result = agent.run("¿Cuánto es 2847 * 1923?")

# Visualización del proceso ReAct
print("\n\n" + "="*70)
print("ESTRUCTURA DEL PATRÓN REACT:")
print("="*70)
print("""
Thought → Action → Observation → Thought → ... → Answer

Ventajas:
  1. Razonamiento explícito y verificable
  2. Puede usar herramientas externas
  3. Auto-verificación de resultados
  4. Transparencia en el proceso de decisión

Casos de uso ideales:
  - Cálculos precisos
  - Búsqueda de información
  - Tareas multi-paso
  - Debugging y verificación
""")
```

---

## Parte 3: RAG - Retrieval Augmented Generation

:::{figure} diagrams/react_agent_loop.png
:name: fig-react-agent-loop
:alt: Bucle del patrón ReAct con Thought, Action, Observation iterativos
:align: center
:width: 90%

**Figura 1:** Patrón ReAct - Thought → Action → Observation loops para razonamiento explícito.
:::

### El Problema

```
Pregunta: "¿Cuáles son los términos de servicio de mi empresa?"

LLM puro:
  No tiene información sobre tu empresa
  Responde genéricamente (inútil)

RAG (Retrieval Augmented Generation):
  1. Busca en tu base de datos de documentos
  2. Recupera términos de servicio relevantes
  3. Incluye en contexto del LLM
  4. LLM responde basado en documento actual
```

### Arquitectura RAG

```{code-cell} ipython3
# Implementación simplificada de RAG (Retrieval Augmented Generation)
import numpy as np
from typing import List, Dict
from dataclasses import dataclass

@dataclass
class Document:
    """Representa un documento en la base de conocimiento"""
    id: str
    content: str
    metadata: Dict = None

class SimpleEmbedding:
    """Embedding simplificado basado en palabras clave"""

    def __init__(self, vocabulary: List[str]):
        self.vocabulary = vocabulary
        self.word_to_idx = {word: i for i, word in enumerate(vocabulary)}

    def embed(self, text: str) -> np.ndarray:
        """Crea un embedding simple basado en frecuencia de palabras"""
        # Embedding de dimensión len(vocabulary)
        embedding = np.zeros(len(self.vocabulary))

        # Contar palabras
        words = text.lower().split()
        for word in words:
            if word in self.word_to_idx:
                embedding[self.word_to_idx[word]] += 1

        # Normalizar
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    def cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calcula similaridad coseno entre dos embeddings"""
        return np.dot(emb1, emb2)

class VectorStore:
    """Almacén vectorial simple para búsqueda semántica"""

    def __init__(self, embedder: SimpleEmbedding):
        self.embedder = embedder
        self.documents: List[Document] = []
        self.embeddings: List[np.ndarray] = []

    def add_document(self, document: Document):
        """Agrega un documento al vector store"""
        embedding = self.embedder.embed(document.content)
        self.documents.append(document)
        self.embeddings.append(embedding)

    def search(self, query: str, top_k: int = 3) -> List[tuple]:
        """Busca documentos similares a la query"""
        query_embedding = self.embedder.embed(query)

        # Calcular similaridades
        similarities = []
        for i, doc_embedding in enumerate(self.embeddings):
            similarity = self.embedder.cosine_similarity(query_embedding, doc_embedding)
            similarities.append((similarity, self.documents[i]))

        # Ordenar por similaridad (mayor a menor)
        similarities.sort(reverse=True, key=lambda x: x[0])

        return similarities[:top_k]

class RAGSystem:
    """Sistema RAG completo"""

    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store

    def augment_prompt(self, query: str, top_k: int = 3) -> str:
        """Recupera contexto relevante y construye prompt aumentado"""
        # Recuperar documentos relevantes
        results = self.vector_store.search(query, top_k)

        # Construir prompt con contexto
        prompt = "Eres un asistente experto. Responde basándote en el contexto.\n\n"
        prompt += "CONTEXTO:\n"

        for i, (similarity, doc) in enumerate(results, 1):
            prompt += f"\n[Documento {i}] (relevancia: {similarity:.3f})\n"
            prompt += f"{doc.content}\n"

        prompt += f"\nPREGUNTA: {query}\n"
        prompt += "\nRESPUESTA: Basándome en el contexto proporcionado, "

        return prompt, results

# Crear vocabulario y embedder
vocabulary = ["términos", "servicio", "política", "privacidad", "datos",
              "usuario", "cuenta", "pago", "cancelación", "reembolso",
              "precio", "suscripción", "contenido", "derechos", "autor"]

embedder = SimpleEmbedding(vocabulary)
vector_store = VectorStore(embedder)

# Agregar documentos de ejemplo
documents = [
    Document(
        id="tos_1",
        content="Los términos de servicio establecen que los usuarios deben "
                "ser mayores de 18 años y aceptar la política de privacidad.",
        metadata={"category": "account"}
    ),
    Document(
        id="tos_2",
        content="La política de cancelación permite reembolsos dentro de 30 días. "
                "Los usuarios pueden cancelar su suscripción en cualquier momento.",
        metadata={"category": "billing"}
    ),
    Document(
        id="tos_3",
        content="Los datos del usuario son protegidos según nuestra política de "
                "privacidad. No compartimos información con terceros.",
        metadata={"category": "privacy"}
    ),
    Document(
        id="tos_4",
        content="El precio de la suscripción es $9.99 al mes. "
                "Se aceptan pagos con tarjeta de crédito.",
        metadata={"category": "billing"}
    ),
    Document(
        id="pricing_1",
        content="El contenido generado por usuarios retiene sus derechos de autor. "
                "La plataforma tiene licencia para mostrar el contenido.",
        metadata={"category": "content"}
    )
]

for doc in documents:
    vector_store.add_document(doc)

print("BASE DE DOCUMENTOS:")
print("="*70)
print(f"Total documentos indexados: {len(vector_store.documents)}")
print(f"Dimensión de embeddings: {len(embedder.vocabulary)}")

# Crear sistema RAG
rag = RAGSystem(vector_store)

# Ejemplos de queries
queries = [
    "¿Cuáles son los términos de servicio?",
    "¿Cómo cancelo mi suscripción?",
    "¿Cuánto cuesta el servicio?"
]

print("\n" + "="*70)
print("EJEMPLOS DE RETRIEVAL:")
print("="*70)

for query in queries:
    print(f"\nQuery: {query}")
    print("-"*70)

    results = vector_store.search(query, top_k=2)

    for i, (similarity, doc) in enumerate(results, 1):
        print(f"\nDocumento {i} (relevancia: {similarity:.3f}):")
        print(f"  ID: {doc.id}")
        print(f"  Contenido: {doc.content[:80]}...")
        print(f"  Categoría: {doc.metadata.get('category', 'N/A')}")

# Demostrar prompt aumentado
print("\n" + "="*70)
print("PROMPT AUMENTADO (RAG):")
print("="*70)

query = "¿Puedo obtener un reembolso?"
augmented_prompt, results = rag.augment_prompt(query, top_k=2)

print(augmented_prompt)

print("\n" + "="*70)
print("VENTAJAS DE RAG:")
print("="*70)
print("""
✓ Información actualizada sin reentrenar el modelo
✓ Reduce alucinaciones (respuestas inventadas)
✓ Respuestas citables (sabemos de dónde vino la info)
✓ Escalable (agregar documentos sin cambiar el modelo)
✓ Específico al dominio

DESVENTAJAS:
✗ Calidad depende del retrieval
✗ Overhead computacional (embeddings + búsqueda)
✗ Requiere mantener documentos actualizados
""")
```

### Ventajas vs Desventajas

```
Ventajas:
  ✓ Información actual sin reentrenamiento
  ✓ Reduce alucinaciones (contextualizado)
  ✓ Citable (puedes señalar de dónde vino la respuesta)

Desventajas:
  ✗ Relevancia de retrieval es crítica (garbage in = garbage out)
  ✗ Overhead computacional (embedding + búsqueda)
  ✗ Requiere mantener base de documentos actualizada
```

---

## Parte 4: Multi-Agent Systems

:::{figure} diagrams/kernelagent_pipeline.png
:name: fig-kernelagent-pipeline
:alt: Pipeline completo de KernelAgent como sistema agéntico multi-componente
:align: center
:width: 100%

**Figura 2:** Arquitectura completa de KernelAgent: los cinco componentes (Prompt Engineering, Constrained Decoding, Tool Use, RAG y Multi-Agent) se integran en un pipeline autónomo capaz de generar, validar y optimizar kernels GPU iterativamente.
:::

Más allá de un agente, ¿qué pasa con múltiples agentes que colaboran?

### Arquitecturas Multi-Agent

#### Tipo 1: Especialistas Colaborativos

```
User: "Analiza este reporte de ventas y genera predicción"

Agente Análisis:
  - Experto en estadística
  - Procesa datos históricos
  - Genera trends

         ↓ (pasa análisis)

Agente Predicción:
  - Experto en ML
  - Usa análisis anterior
  - Genera forecast

         ↓ (pasa predicción)

Agente Presentación:
  - Experto en comunicación
  - Formatea resultado
  - Retorna a usuario
```

#### Tipo 2: Crítica y Mejora Iterativa

```
Agente Propuesta:
  Genera primera versión de solución

         ↓

Agente Crítica:
  Evalúa, señala problemas

         ↓

Agente Propuesta (v2):
  Itera basado en crítica

         ↓ (repite hasta bueno)

Usuario obtiene solución mejorada
```

#### Tipo 3: Competencia y Votación

```
3 Agentes generan soluciones diferentes

Agente 1: Propuesta A
Agente 2: Propuesta B
Agente 3: Propuesta C

Agente Evaluador:
  Compara las 3 propuestas
  Elige la mejor
  O combina lo mejor de cada una
```

### Ejemplo Práctico: Sistema de Análisis de Datos

```{code-cell} ipython3
# Sistema Multi-Agente para Análisis de Datos
from typing import List, Dict, Any
from dataclasses import dataclass
import random

@dataclass
class AgentMessage:
    """Mensaje entre agentes"""
    sender: str
    receiver: str
    content: str
    timestamp: str

class Agent:
    """Agente especializado en una tarea"""

    def __init__(self, name: str, expertise: str):
        self.name = name
        self.expertise = expertise
        self.message_history: List[AgentMessage] = []

    def run(self, task: str, context: str = "") -> str:
        """Ejecuta una tarea (simulado)"""
        print(f"\n[{self.name}] Processing task...")
        print(f"  Expertise: {self.expertise}")
        print(f"  Task: {task[:60]}...")

        # Simular procesamiento
        result = self._simulate_task(task, context)
        return result

    def _simulate_task(self, task: str, context: str) -> str:
        """Simula ejecución de tarea"""
        # Para demostración, generamos resultados simulados
        if "explora" in task.lower():
            return self._explore_data()
        elif "analiza" in task.lower() or "calcula" in task.lower():
            return self._analyze_data(context)
        elif "reporte" in task.lower() or "reporta" in task.lower():
            return self._create_report(context)
        else:
            return f"Resultado de la tarea: {task}"

    def _explore_data(self) -> str:
        """Simula exploración de datos"""
        return """
EXPLORACIÓN DE DATOS:
- Tipo de datos: CSV con 10,000 filas, 8 columnas
- Columnas: fecha, producto, cantidad, precio, región, vendedor, categoría, descuento
- Missing values: 2% en columna 'descuento'
- Distribución: Ventas concentradas en Q4 (40%)
- Rango de precios: $10 - $500
- Categorías: 5 (Electrónica, Ropa, Hogar, Deportes, Libros)
"""

    def _analyze_data(self, exploration: str) -> str:
        """Simula análisis estadístico"""
        return """
ANÁLISIS ESTADÍSTICO:
- Correlación precio-cantidad: -0.65 (negativa moderada)
- Tendencia temporal: Crecimiento 15% anual
- Anomalías detectadas: 3 ventas atípicas (>3σ)
- Región con mejor desempeño: Norte (35% del total)
- Producto estrella: Laptop Pro ($250K en ventas)
- Estacionalidad: Picos en Noviembre y Diciembre
"""

    def _create_report(self, analysis: str) -> str:
        """Simula creación de reporte"""
        return """
REPORTE EJECUTIVO - ANÁLISIS DE VENTAS 2024

RESUMEN:
Las ventas alcanzaron $2.5M con un crecimiento del 15% anual.
Se observa una fuerte estacionalidad en Q4 y una correlación
negativa entre precio y cantidad vendida.

HALLAZGOS CLAVE:
1. Región Norte lidera con 35% de las ventas totales
2. Producto estrella: Laptop Pro ($250K)
3. Concentración del 40% de ventas en Q4

RECOMENDACIONES:
- Aumentar inventario en Q4
- Expandir estrategia en Región Norte
- Investigar productos premium de bajo volumen
"""

class MultiAgentSystem:
    """Sistema multi-agente colaborativo"""

    def __init__(self):
        self.agents: Dict[str, Agent] = {}
        self.communication_log: List[AgentMessage] = []

    def register_agent(self, agent: Agent):
        """Registra un agente en el sistema"""
        self.agents[agent.name] = agent
        print(f"Registered agent: {agent.name} ({agent.expertise})")

    def send_message(self, sender: str, receiver: str, content: str):
        """Envía mensaje entre agentes"""
        message = AgentMessage(
            sender=sender,
            receiver=receiver,
            content=content,
            timestamp="2024-01-01T10:00:00"
        )
        self.communication_log.append(message)

    def run_pipeline(self, agents_sequence: List[str], initial_task: str) -> str:
        """Ejecuta un pipeline de agentes secuencialmente"""
        current_output = initial_task
        results = {}

        print("\n" + "="*70)
        print("MULTI-AGENT PIPELINE EXECUTION")
        print("="*70)

        for i, agent_name in enumerate(agents_sequence):
            agent = self.agents[agent_name]

            # Contexto es la salida del agente anterior
            context = current_output if i > 0 else ""

            # Ejecutar agente
            result = agent.run(current_output, context)
            results[agent_name] = result

            # Guardar para siguiente agente
            current_output = result

            # Log comunicación
            if i < len(agents_sequence) - 1:
                next_agent = agents_sequence[i + 1]
                self.send_message(agent_name, next_agent, result[:100] + "...")

        return current_output

# Crear sistema multi-agente
system = MultiAgentSystem()

# Crear agentes especializados
data_agent = Agent(
    name="DataExplorer",
    expertise="Exploración y limpieza de datos"
)

analysis_agent = Agent(
    name="StatisticalAnalyzer",
    expertise="Análisis estadístico y correlaciones"
)

report_agent = Agent(
    name="ReportGenerator",
    expertise="Generación de reportes ejecutivos"
)

# Registrar agentes
system.register_agent(data_agent)
system.register_agent(analysis_agent)
system.register_agent(report_agent)

# Ejecutar pipeline
print("\n" + "="*70)
print("TASK: Analizar dataset de ventas 2024")
print("="*70)

pipeline = ["DataExplorer", "StatisticalAnalyzer", "ReportGenerator"]
initial_task = "Explora sales_2024.csv: tipos, distribuciones, missing values"

final_report = system.run_pipeline(pipeline, initial_task)

print("\n" + "="*70)
print("RESULTADO FINAL:")
print("="*70)
print(final_report)

# Visualizar arquitectura multi-agente
print("\n" + "="*70)
print("ARQUITECTURA MULTI-AGENTE:")
print("="*70)
print("""
User Request
     ↓
DataExplorer (Agente 1)
     ↓ [datos explorados]
StatisticalAnalyzer (Agente 2)
     ↓ [análisis estadístico]
ReportGenerator (Agente 3)
     ↓
Final Report → User

VENTAJAS:
✓ Especialización: cada agente experto en su dominio
✓ Modularidad: fácil agregar/reemplazar agentes
✓ Escalabilidad: paralelizar agentes independientes
✓ Trazabilidad: log completo de comunicación

TIPOS DE ARQUITECTURAS:
1. Pipeline (secuencial): A → B → C
2. Crítica iterativa: Propuesta ⇄ Revisor
3. Competencia: múltiples agentes compiten
4. Colaborativa: agentes negocian solución
""")

# Mostrar log de comunicación
print("\n" + "="*70)
print("LOG DE COMUNICACIÓN:")
print("="*70)
for msg in system.communication_log:
    print(f"{msg.sender} → {msg.receiver}:")
    print(f"  {msg.content}")
    print()
```

---

## Parte 5: Planning y Reasoning Avanzado

Más allá de ReAct, cómo pueden los agentes planificar tareas complejas:

### Chain-of-Thought Extendido

```
Pregunta: "Planifica mi viaje a Francia por 1 semana"

LLM:

**Breakdown:**
1. Decidir fechas (7 días)
2. Reservar vuelo
3. Reservar hotel
4. Planificar itinerario (museos, restaurantes)
5. Preparar documentos (pasaporte, dinero)

**Subproblemas:**
1.1 ¿Cuándo viajas? (necesito fecha del usuario)
1.2 ¿Cuál es tu presupuesto?
1.3 ¿Qué te interesa? (arte, comida, naturaleza)

**Acciones inmediatas:**
- Pregunta al usuario sobre preferencias

**Acciones después:**
- Busca vuelos
- Busca hoteles
- Genera itinerario

...

Ejecuta este plan paso a paso
```

### Programación Neuro-Simbólica

Combina LLMs (flexible) con lógica simbólica (rigurosa):

```
LLM decide: "Este email debe archivarse"
Sistema simbólico verifica: "¿Regla de negocio autoriza archivar?"
Si no: Retorna al LLM: "No puedo, violaria política X"
LLM propone: "¿Lo marco como spam entonces?"
Sistema simbólico verifica: OK
Ejecuta acción
```

---

## Parte 6: Futura de IA Estructurada

### El Horizonte

```
Hoy (2024):
  - LLMs generan texto
  - Tool use es manual (requiere definir funciones)
  - Razonamiento es implícito (emergente)

Próximo (2025-2026):
  - LLMs descubren herramientas automáticamente
  - Razonamiento es explícito (trazable)
  - Multi-agent es estándar

Futuro (2027+):
  - IA que puede verificar su propio razonamiento
  - Auto-improvement (mejora sin feedback humano)
  - Razonamiento causal (no solo correlación)
```

### Tendencias

```{code-cell} ipython3
# Visualización de tendencias en sistemas agénticos
import matplotlib.pyplot as plt
import numpy as np

# Datos de adopción de diferentes paradigmas (simulados)
years = np.array([2020, 2021, 2022, 2023, 2024, 2025, 2026])
paradigms = {
    'Monolithic LLMs': [10, 25, 50, 70, 75, 70, 60],
    'Tool Use / Function Calling': [5, 10, 30, 60, 80, 90, 95],
    'RAG Systems': [2, 8, 25, 50, 75, 85, 90],
    'Multi-Agent Systems': [1, 3, 10, 25, 45, 70, 85],
    'Verified Reasoning': [0, 1, 3, 8, 20, 40, 65]
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Gráfico 1: Adopción de paradigmas
for paradigm, adoption in paradigms.items():
    ax1.plot(years, adoption, marker='o', linewidth=2, label=paradigm, markersize=6)

ax1.set_xlabel('Año', fontsize=11)
ax1.set_ylabel('Adopción en Producción (%)', fontsize=11)
ax1.set_title('Evolución de Paradigmas en IA', fontsize=12, fontweight='bold')
ax1.legend(loc='best', fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0, 105])

# Gráfico 2: Comparación de características
characteristics = ['Compositionality', 'Transparency', 'Robustness',
                  'Efficiency', 'Interactivity']

current_2024 = [45, 40, 35, 60, 50]
future_2026 = [75, 70, 65, 80, 85]

x = np.arange(len(characteristics))
width = 0.35

bars1 = ax2.barh(x - width/2, current_2024, width, label='2024 (Actual)',
                 alpha=0.8, color='#3498db')
bars2 = ax2.barh(x + width/2, future_2026, width, label='2026 (Proyectado)',
                 alpha=0.8, color='#2ecc71')

ax2.set_xlabel('Madurez (%)', fontsize=11)
ax2.set_title('Madurez de Características Clave', fontsize=12, fontweight='bold')
ax2.set_yticks(x)
ax2.set_yticklabels(characteristics)
ax2.legend()
ax2.grid(True, alpha=0.3, axis='x')
ax2.set_xlim([0, 100])

# Añadir valores en las barras
for bars in [bars1, bars2]:
    for bar in bars:
        width = bar.get_width()
        ax2.text(width + 2, bar.get_y() + bar.get_height()/2,
                f'{int(width)}%', ha='left', va='center', fontsize=9)

plt.tight_layout()
plt.show()

print("TENDENCIAS EN SISTEMAS AGÉNTICOS")
print("="*70)
print("""
1. COMPOSITIONALITY (Componibilidad)
   Estado actual: Emergente
   Futuro: Estándar de la industria
   → Armar sistemas complejos de piezas simples y verificables
   → Ejemplo: Combinar múltiples agentes especializados

2. TRANSPARENCY (Transparencia)
   Estado actual: Limitada
   Futuro: Razonamiento trazable y explicable
   → Entender cómo razona la IA (explicabilidad)
   → Ejemplo: Auditoría completa de decisiones

3. ROBUSTNESS (Robustez)
   Estado actual: Frágil fuera de distribución
   Futuro: Sistemas adaptativos y resistentes
   → No fallan con inputs inesperados
   → Ejemplo: Graceful degradation cuando faltan herramientas

4. EFFICIENCY (Eficiencia)
   Estado actual: Modelos grandes y costosos
   Futuro: Modelos pequeños especializados
   → Hacer más con menos parámetros
   → Ejemplo: MoE (Mixture of Experts), destilación

5. INTERACTIVITY (Interactividad)
   Estado actual: Loops humano-en-el-loop básicos
   Futuro: Colaboración fluida humano-IA
   → Humanos y IA colaborando, no sustituyendo
   → Ejemplo: Co-creación en tiempo real
""")

# Proyección de capacidades
print("\n" + "="*70)
print("PROYECCIÓN DE CAPACIDADES (2024 → 2027+)")
print("="*70)

timeline = {
    "2024 (Hoy)": [
        "LLMs generan texto",
        "Tool use es manual (requiere definir funciones)",
        "Razonamiento implícito (emergente)",
        "Multi-agente experimental"
    ],
    "2025-2026 (Próximo)": [
        "LLMs descubren herramientas automáticamente",
        "Razonamiento explícito (trazable)",
        "Multi-agente es estándar",
        "Verificación básica de outputs"
    ],
    "2027+ (Futuro)": [
        "IA verifica su propio razonamiento",
        "Auto-improvement (mejora sin feedback humano)",
        "Razonamiento causal (no solo correlación)",
        "Sistemas completamente autónomos"
    ]
}

for period, capabilities in timeline.items():
    print(f"\n{period}:")
    for cap in capabilities:
        print(f"  • {cap}")
```

---

## Parte 7: Ejemplo Completo: Asistente de Investigación

```python
class ResearchAssistant:
    """Agente que investiga preguntas complejas"""

    def __init__(self):
        self.tools = {
            "search_web": search_web,
            "fetch_paper": fetch_academic_paper,
            "summarize": summarize_text,
            "calculate": calculate
        }

    def research(self, query):
        """Investiga una pregunta"""

        # Fase 1: Planning (ReAct)
        plan = self.llm.generate(f"""
            Pregunta: {query}

            Tareas a realizar (ordena por importancia):
            1. ...
            2. ...
        """)

        # Fase 2: Ejecución (Multi-step)
        results = []
        for task in plan.tasks:
            result = self.execute_task(task)
            results.append(result)

        # Fase 3: Síntesis (RAG)
        context = "\n".join(results)
        final_answer = self.llm.generate(f"""
            Basándote en esta investigación:
            {context}

            Responde la pregunta original: {query}
            Include fuentes.
        """)

        return final_answer

    def execute_task(self, task):
        """Ejecuta una tarea individual"""
        # Determina qué herramientas necesita
        tools_needed = self.llm.generate(f"""
            Tarea: {task}
            Herramientas disponibles: {list(self.tools.keys())}

            ¿Cuáles necesitas?
        """)

        # Ejecuta herramientas
        results = []
        for tool in tools_needed:
            result = self.tools[tool](task)
            results.append(result)

        return "\n".join(results)

# Uso
assistant = ResearchAssistant()
answer = assistant.research("¿Cuáles son los avances recientes en fusion nuclear?")
print(answer)
```

---

## Reflexión y Ejercicios

### Preguntas para Reflexionar:

1. **Tool Use vs Fine-Tuning:** ¿Cuándo es mejor enseñar a un LLM con fine-tuning vs darle acceso a herramientas?

2. **RAG vs Knowledge Update:** ¿Por qué RAG es mejor que reentrenar el modelo cada semana?

3. **Multi-Agent:** ¿Cuándo agregar un segundo agente mejora el sistema vs solo agrega complejidad?

### Ejercicios Prácticos:

1. **Diseña función calling:**
   ```
   Aplicación: Asistente bancario que ayuda a transferencias

   Define 5 funciones con schemas JSON:
   1. get_account_balance
   2. transfer_funds
   3. check_transaction_history
   4. ...

   Para cada función, especifica:
   - Descripción
   - Parámetros con tipos
   - Restricciones de seguridad
   ```

2. **RAG Pipeline:**
   ```
   Tienes 100 documentos técnicos (PDFs).
   Diseña pipeline:
   1. Cómo extraes texto de PDFs?
   2. Cómo generas embeddings?
   3. Qué vector store usas?
   4. Cómo evalúas calidad de retrieval?
   ```

3. **Multi-Agent Collaboration:**
   ```
   Tarea: "Analiza rentabilidad de proyecto de energía solar"

   Define 3 agentes especializados:
   1. Agente técnico
   2. Agente financiero
   3. Agente ambiental

   Para cada uno, describe:
   - Expertise específica
   - Herramientas que usa
   - Cómo colaboran con otros agentes
   ```

4. **Reflexión escrita (400 palabras):** "El futuro será sistemas agénticos con múltiples LLMs especializados colaborando. ¿Cómo cambiaría esto la forma en que desarrolladores construyen aplicaciones? ¿Qué nuevos desafíos surgirían?"

---

## Puntos Clave

- **Tool Use:** LLM decide qué herramientas usar; acceso a información actual y acciones externas
- **ReAct:** Razonamiento explícito (Thought) + Acción + Observación
- **RAG:** Recupera documentos relevantes antes de generar; reduce alucinaciones
- **Multi-Agent:** Múltiples agentes especializados colaboran para resolver problemas complejos
- **Planning:** LLM descompone tareas complejas en subtareas manejables
- **Compositionality:** Sistemas complejos de piezas simples y verificables
- **Futuro:** IA transparente, robusta, eficiente; colaboración humano-IA

```{admonition} Resumen
:class: important
**Lo que aprendiste:**
- Tool use / function calling permite a LLMs acceder a información actual (APIs, bases de datos) y ejecutar acciones (enviar emails, actualizar sistemas)
- ReAct (Reasoning + Acting) hace el razonamiento explícito: Thought → Action → Observation → repetir
- RAG reduce alucinaciones recuperando documentos relevantes antes de generar, actualizando conocimiento sin reentrenamiento
- Multi-agent systems con especialización (DataExplorer, Analyzer, Reporter) dividen tareas complejas en piezas manejables
- Futuro: compositionality (sistemas de componentes verificables), transparency (razonamiento trazable), robustness (adaptativos)

**Conclusión del módulo:** Has completado el journey desde fundamentos teóricos hasta sistemas prácticos de producción. Ahora puedes construir KernelAgent como sistema agéntico completo: prompt engineering + constrained decoding + tool use + RAG + multi-agent. El futuro no es modelos más grandes, sino sistemas más inteligentes.
```

```{admonition} Verifica tu comprensión
:class: note
1. Diseña 3 tools para KernelAgent: a) compile_triton, b) run_benchmark, c) search_docs. Especifica sus schemas JSON
2. ¿Cómo implementarías ReAct en KernelAgent para mejorar iterativamente un kernel que falla benchmarks?
3. Para RAG, ¿qué documentos indexarías? (Triton docs, ejemplos, papers de optimización)
4. Arquitecta un sistema multi-agente para KernelAgent: ProposerAgent, CriticAgent, OptimizerAgent. ¿Cómo colaboran?
```

---

## Conclusión de la Serie

En estas 10 lecturas, viajamos desde los fundamentos de IA clásica hasta sistemas agénticos modernos:

1. Paradigmas históricos y teóricos
2. Matemáticas de redes neuronales
3. Arquitectura Transformer
4. Generación autoregresiva
5. Muestreo y restricciones
6. LLMs para código
7. Infraestructura de serving
8. Economía y costos
9. Mejora y evaluación
10. Sistemas agénticos

El futuro de IA no es un modelo más grande, sino **sistemas más inteligentes**: razonamiento transparente, herramientas integradas, colaboración entre agentes y humanos.

Tu rol como ingeniero es entender estos componentes, elegir las herramientas correctas, y construir sistemas que sean efectivos, eficientes y confiables.

Adelante.

---

## Referencias

- Yao, S. et al. (2023). [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629). ICLR 2023.
- Lewis, P. et al. (2020). [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401). NeurIPS 2020.
- Schick, T. et al. (2023). [Toolformer: Language Models Can Teach Themselves to Use Tools](https://arxiv.org/abs/2302.04761). arXiv.

