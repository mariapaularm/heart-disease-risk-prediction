# -Laboratorio-3-DOSW
# Integrantes
Juan Diego Rodriguez Velasquez
Daniel Felipe Hueso
Maria Paula Rodriguez Muñoz

Reto #1)

Reglas de negocios a tener en cuenta: 
Los números de cuenta deben tener exactamente 10 dígitos.
Una cuenta solo es válida si sus 2 primeros dígitos corresponden a un banco registrado.
Los números de cuenta no pueden contener letras ni caracteres especiales, solo números.
Solo se pueden realizar operaciones sobre cuentas válidas y registradas en el sistema.
Los depósitos deben aumentar el saldo de la cuenta de manera correcta.
Las consultas de saldo deben mostrar la información actualizada.

Las principales funcionalidades serian : 

Crear una cuenta bancaria .
Validar el numero de cuenta .
Consultar saldo de la cuenta .
Realizar depósitos .

Precondiciones del sistema serian : 
Debe existir un listado de bancos registrados
El sistema debe tener una base de datos activa para almacenar cuenta y saldos. 
El cliente debe registrar una cuenta valida antes de poder consultar el saldo o hacer depósitos.
El sistema debe garantizar la seguridad y validación de datos (no puedfe tener duplicados ni cuentas invalidas)

Reto #2

1 ) Explicacion diagrama de contexto : 

Cliente/Usuario: actor que interactúa con la interfaz (web o API) para crear cuentas, consultar saldo y depositar.

Interfaz (UI/API): puerta de entrada que valida entradas simples y reenvía las solicitudes al Sistema Bankify.

Sistema Bankify: nucleo que aplica reglas de negocio (10 dígitos, codigos de banco, solo digitos), 
gestiona saldos y registra transacciones.

Base de datos: almacena cuentas, saldos y logs de operaciones.

Bancos registrados: fuente de verdad para los códigos de bancos 
El sistema puede tener una tabla local con esos códigos o consultar un servicio externo.

2)Diagrama de casos de uso en uml:

<img width="910" height="694" alt="image" src="https://github.com/user-attachments/assets/fdb47683-03b9-4035-a27a-abbb3f39b0ae" />


3)Historias de usuario:

Como Cliente, quiero crear una cuenta bancaria proporcionando un numero de cuenta válido,
para poder guardar mi cuenta en Bankify y operar con ella.

Como Cliente, quiero que el sistema valide que el numero de cuenta tenga exactamente 10 digitos y
que los 2 primeros correspondan a un banco registrado, para asegurar que la cuenta es válida antes de crearla.

Como Cliente, quiero consultar el saldo de mi cuenta, para conocer cuánto dinero tengo disponible.

Como Cliente, quiero depositar dinero en mi cuenta, para aumentar mi saldo y poder realizar operaciones futuras.

Como Administrador del sistema, quiero mantener la lista de bancos registrados 
(códigos de 2 dígitos), para que las validaciones de cuentas sean correctas y actualizables.

Como Cliente, quiero recibir mensajes de error claros si intento operar con una cuenta invalida,
para corregir los datos y volver a intentar.

4)Historia de usuario archivo Exel : 

<img width="906" height="479" alt="image" src="https://github.com/user-attachments/assets/3a012eaa-7a0a-443a-93a2-8c1c1a06a7c0" />


5) Diagrama de clases:

<img width="1099" height="581" alt="image" src="https://github.com/user-attachments/assets/01c34fb9-fc4b-4a37-906b-094b16710f3a" />






# heart-disease-risk-prediction
