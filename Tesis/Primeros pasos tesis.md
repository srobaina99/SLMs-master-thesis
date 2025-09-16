1. leer Paper tiny stories
2. Leer paper mechanistic interpretability (ver si tocan los weights)
3. Tocar los weights
# Segunda semana

# Primer semana
- Leer paper de tiny stories:
	- Entrenan modelos de lenguaje crudos con historias generadas (2.12M rows) por un modelo polenta
	- Idea interesante: 
	  Se podría pensar en generar bastante data sintética que emule conversaciones con típicos errores de estudiantes (hechos por profesores) y correcciones. Y entrenar modelos chicos que puedan correr en maquinas locales,
- Empecé a tratar de correr local algún modelo chico pero no llegué a mucho aún

#TODO
- [x] Planilla de la Modelos, dataset, URL, info 
- [ ] Buscar modelos chicos para finetunear (tipo qwen)
- [ ] Seguir entrando a SLM
- [ ] Pensar en formas de generar data sintética

- [ ] Tocar pesos de modelos más grandes
	- [ ] En local no puedo correr los modelos más chiquitos comerciales (tipo tiny lama)
- [ ] Prototipo de agente conversacional con modelos chiquitos

# Tercer semana
Ver como mechar la idea de controlar las salidas

Objetivo en principio de la tesis:
El nivel de lenguaje del chatbot es apropiado para el nivel del estudiante?
Posibilidades: 
- Acotar los datos de entrenamiento
- Tener un modelo chico atento a la dificultad del lenguaje

Para primer semana de Julio presentar paper: Small models complementando BEA https://arxiv.org/pdf/2501.05465v1

#### Próximos pasos:
- [x] Experimentos con tinystories ( con textos nuestros )
	- Los instruct de tinystories no son chat, son para dar instrucciones de cuentos
- [ ] Pensar experimentos

# 5ta semana
- [ ] Probar qwen con prompts más largos discutiendo una historia caso de uso cómo en el proyecto

Definir grado de complejidad de textos en inglés
-> leer trabajo de grado que evalúan eurísticas para simplificar inglés (buscar en el mail)
-> Buscar vocabulario esperado de inglés para x edad (buscar en el mail)

-> Volver a pensar esto de pesar distinto las palabras del vocabulario que queremos. EMPEZAR POR EL PROMPT (si alcanza no valdría la pena)

- **Estandarizar un poco más los experimentos (por ejemplo con las métricas de dificultad)**
---
Presentación del review de modelos chicos
Cómo se generan los modelos chicos, estrategias y demás
| Para primer semana de Julio presentar paper: Small models complementando BEA https://arxiv.org/pdf/2501.05465v1 |

## TODO
- [ ] buscar vocabulario esperado en mail
	- [ ] Ordenar esos archivos, armar un drive
- [x] informe de grado en mail
- [ ] Armar presentación de paper para el grupo ⚠️⚠️


# 7ma Semana
- [ ] Ordenar la bibliografía, tener la data de los papers y los links
### Ideas para probar
- Algo medio agéntico, herramientas concretas llevadas acabo por pequeños modelos
	- Pensar tools para el agente, por ahí alguna herramienta que tenga 


# 23/07

- [x] Probar Qwen 0.5
- [x] Probar agregar la lista de palabras simples a los weighted 



Lecturas dirigidas para juntar créditos. 
	Probar experimento chico y hacer un informecito 
	Leer paper de speech2text


# 14/08
- [ ] sistematizar un poco para evaluar
	- [ ] Generar varias salidas con un mismo prompt con y sin pesos
	- [ ] Evaluar comparativamente
		- Correcto
		- Sencillez (pensar)
		- tiempo
		- indice de dificultad
	- [ ] Definir el prompt al principio
- [ ] Debuggear por que eligió princess cuando no estaba en el top 10