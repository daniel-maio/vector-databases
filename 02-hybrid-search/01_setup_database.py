# Módulo de Configuração do Banco de Dados SQL

import sqlite3

def cria_banco_suporte():
    
    connection = sqlite3.connect('suporte_tecnico.db')
    cursor = connection.cursor()
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS artigos_suporte (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                titulo TEXT NOT NULL,
                conteudo TEXT NOT NULL,
                categoria TEXT,
                data_atualizacao TEXT
    )
    ''')

    # Dados de exemplo (Casos difíceis para busca puramente vetorial ou puramente keyword)
    dados = [
        ("Erro 503 no Gateway de Pagamento", "Se o gateway retornar 503, verifique se o balanceador de carga Nginx está com a configuração de timeout correta. Reinicie o serviço com 'systemctl restart nginx'.", "Infraestrutura", "2026-01-10"),
        ("Reset de Senha de Usuário Admin", "Para resetar a senha de admin, acesse o painel de controle, vá em Configurações > Segurança e clique em 'Forçar Logout'. O link de reset será enviado ao email cadastrado.", "Acesso", "2025-12-05"),
        ("Python: ModuleNotFound Error", "Esse erro ocorre quando a biblioteca não está instalada no ambiente virtual (venv). Execute 'pip install -r requirements.txt' e verifique se o venv está ativo.", "Desenvolvimento", "2024-02-20"),
        ("Lentidão na Consulta SQL", "Queries lentas geralmente indicam falta de índices. Use o comando EXPLAIN ANALYZE para verificar o plano de execução. Evite SELECT * em tabelas grandes.", "Banco de Dados", "2024-01-15"),
        ("Configuração de VPN Corporativa", "Para acessar a VPN, utilize o cliente Cisco AnyConnect. O endereço do servidor é vpn.empresa.com. O protocolo utilizado é DTLS para maior performance.", "Rede", "2023-11-30")
    ]

    cursor.executemany('''
    INSERT INTO artigos_suporte (titulo, conteudo, categoria, data_atualizacao)
    VALUES (?, ?, ?, ?)
    ''',dados)

    # Grava
    connection.commit()
    print(f"Banco de dados SQL criado com {len(dados)} artigos.")

    # Fecha a conexão
    connection.close()


if __name__ == "__main__":
    cria_banco_suporte()




    

