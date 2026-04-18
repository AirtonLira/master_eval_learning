"""
hr_documents.py
---------------
Base de conhecimento de RH — documentos que alimentam o RAG.
 
Em produção, isso seria um PDF ou banco de dados real.
Aqui usamos texto direto para focar no eval, não na ingestão.
"""

HR_DOCUMENTS = [
    {
        "id": "ferias-001",
        "title": "Política de Férias",
        "content": """
        Férias Anuais
        Todo colaborador tem direito a 30 dias corridos de férias após completar
        12 meses de trabalho (período aquisitivo). As férias podem ser parceladas
        em até 3 períodos, sendo que um deles não pode ser inferior a 14 dias
        corridos e os demais não podem ser inferiores a 5 dias corridos cada.
        O pagamento das férias deve ser realizado até 2 dias antes do início
        do período de descanso, com acréscimo de 1/3 constitucional.
        Férias coletivas podem ser concedidas pela empresa em períodos de baixa
        demanda, com aviso prévio de 15 dias.
        """,
    },
    {
        "id": "plano-saude-001",
        "title": "Plano de Saúde e Odontológico",
        "content": """
        Plano de Saúde
        A empresa oferece plano de saúde Bradesco Saúde (plano Flex Nacional)
        para todos os colaboradores a partir do primeiro dia de trabalho.
        O custo é compartilhado: a empresa cobre 80% da mensalidade do titular
        e 60% dos dependentes diretos (cônjuge e filhos até 21 anos ou 24 anos
        se universitários). Coparticipação de 20% em consultas e 30% em exames.
 
        Plano Odontológico
        Plano Bradesco Dental disponível opcionalmente. A empresa cobre 70%
        do custo do titular. Inclui consultas, limpeza, restaurações e ortodontia
        básica sem coparticipação.
        """,
    },
    {
        "id": "home-office-001",
        "title": "Política de Home Office",
        "content": """
        Trabalho Remoto
        Colaboradores em regime híbrido trabalham presencialmente 3 dias por semana
        (terça, quarta e quinta), com home office às segundas e sextas.
        A empresa fornece auxílio home office de R$ 150 por mês para cobrir
        custos de internet e energia elétrica, pago junto com o salário.
        Colaboradores full remote (aprovados pelo gestor e RH) recebem R$ 300
        por mês de auxílio. Equipamentos (notebook, monitor, teclado) são
        fornecidos pela empresa e devolvidos no desligamento.
        """,
    },
    {
        "id": "rescisao-001",
        "title": "Política de Rescisão e Aviso Prévio",
        "content": """
        Aviso Prévio
        O aviso prévio é de 30 dias para colaboradores com até 1 ano de empresa.
        Para cada ano completo acima do primeiro, acrescentam-se 3 dias ao aviso,
        com limite máximo de 90 dias. O colaborador pode trabalhar o aviso ou
        receber indenizado (a critério da empresa em caso de demissão sem justa causa).
 
        Verbas Rescisórias
        Em demissão sem justa causa: saldo de salário, férias vencidas + 1/3,
        férias proporcionais + 1/3, 13º proporcional, multa de 40% do FGTS
        e saque do FGTS. Prazo de pagamento: 10 dias corridos após o término
        do contrato.
        """,
    },
    {
        "id": "beneficios-001",
        "title": "Benefícios Gerais",
        "content": """
        Vale Refeição e Alimentação
        Vale Refeição: R$ 35 por dia útil trabalhado (cartão Flash).
        Vale Alimentação: R$ 600 por mês, creditado todo dia 1º.
        Ambos são isentos de desconto em folha.
 
        Vale Transporte
        Fornecido conforme necessidade e declaração de rota do colaborador.
        Desconto em folha de 6% do salário bruto ou o valor do benefício,
        o que for menor.
 
        Seguro de Vida
        Seguro de vida em grupo para todos os colaboradores, sem custo.
        Cobertura de R$ 200.000 em caso de morte natural ou acidental.
 
        Gympass
        Acesso a academias e estúdios parceiros. Planos a partir de R$ 29,90/mês
        com subsídio de 50% da empresa.
        """,
    },
    {
        "id": "desenvolvimento-001",
        "title": "Desenvolvimento e Capacitação",
        "content": """
        Bolsa Educação
        A empresa reembolsa até R$ 800 por mês em cursos relacionados à função
        do colaborador (graduação, pós, cursos técnicos, idiomas). Requer aprovação
        prévia do gestor e permanência mínima de 6 meses após o término do curso.
 
        Budget de Treinamento
        Cada área tem budget anual de R$ 3.000 por colaborador para treinamentos
        e conferências. Aprovação via gestor direto.
 
        Plano de Carreira
        Revisões de cargo e salário acontecem anualmente em março. Promoções
        fora do ciclo são possíveis em casos excepcionais com aprovação do C-level.
        """,
    },
    {
        "id": "licencas-001",
        "title": "Licenças e Afastamentos",
        "content": """
        Licença Maternidade
        180 dias de licença maternidade remunerada (30 dias a mais que o mínimo
        legal), sem desconto em folha.
 
        Licença Paternidade
        20 dias de licença paternidade remunerada (15 dias a mais que o mínimo legal).
 
        Licença por Luto
        3 dias úteis em caso de falecimento de cônjuge, filhos ou pais.
        1 dia útil para falecimento de irmãos, sogros e avós.
 
        Day Off de Aniversário
        Colaboradores têm direito a 1 dia de folga no mês do aniversário,
        a ser agendado com o gestor com antecedência mínima de 3 dias.
        """,
    },
]