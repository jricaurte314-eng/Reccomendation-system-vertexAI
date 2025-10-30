``` mermaid
flowchart TD
  START([Inicio]) --> A1

  %% Fase A - Recoleccion
  subgraph FA["Fase A - Recoleccion"]
    A1[Buscar documentos clave<br/>y datos necesarios]
    A2{Hay informacion suficiente}
    A1 --> A2
    A2 -- No --> A1
  end
  A2 -- Si --> B1

  %% Fase B - Redaccion
  subgraph FB["Fase B - Redaccion"]
    B1[Agente: Crear varios borradores]
    B2[Agente: Revisar claridad, fuentes y dar puntaje]
    B3[Mejorar lo mas debil]
    B4{Borrador listo}
    B1 --> B2 --> B3 --> B4
    B4 -- No --> B1
  end
  B4 -- Si --> C1

  %% Fase C - Revision y mejoras
  subgraph FC["Fase C - Revision y mejoras"]
    C1[Verificar estructura y requisitos minimos]
    C2[Agente: Hacer claro y medible el texto]
    C3[Agente: Comprobar reglas legales y no discriminacion]
    C4[Agente: Confirmar que cada punto tenga su fuente]
    C5[Agente: Revisar coherencia con el cargo]
    C6{Todo correcto}
    C1 --> C2 --> C3 --> C4 --> C5 --> C6

    %% Mini mejoras
    subgraph CM["Mini ciclo de mejoras"]
      CM1[Proponer ajustes puntuales]
      CM2[Volver a revisar el resultado]
      CM1 --> CM2
    end
    C6 -- No --> CM1
    CM2 --> C2
  end
  C6 -- Si --> D1

  %% Fase D - Evaluacion y decision
  subgraph FD["Fase D - Evaluacion y decision"]
    D1[Calcular puntaje de calidad]
    D2[Calculo de Indice de confiabilidad IC con factores relevantes]
    D3{Puntaje suficiente}
    D1 --> D2 --> D3
  end

  D3 -- "IC >= 0.80 - Alto" --> E1[Revision humana final]
  D3 -- "0.60 <= IC < 0.80 - Medio" --> E2[Pequeños ajustes]
  D3 -- "0.60 < IC - Bajo" --> B1
  E2 --> CM1

  E1 --> F1[Guardar version aprobada Golden]
  F1 --> F2[Mejora continua y lecciones]
  F2 --> END([Fin])

  %% Estilos simples por fase
  classDef phaseA fill:#cce5ff,stroke:#2f6fad,color:#0b1e39;
  classDef phaseB fill:#d4edda,stroke:#2f7d32,color:#0a2912;
  classDef phaseC fill:#fff3cd,stroke:#ad8b24,color:#3b2a06;
  classDef phaseD fill:#f8d7da,stroke:#a1262d,color:#3a0b0f;
  classDef phaseE fill:#e2e3e5,stroke:#6c757d,color:#23272b;

  class A1,A2 phaseA
  class B1,B2,B3,B4 phaseB
  class C1,C2,C3,C4,C5,C6,CM1,CM2 phaseC
  class D1,D2,D3 phaseD
  class E1,E2,F1,F2,START,END phaseE


```
