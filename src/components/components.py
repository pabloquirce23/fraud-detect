from static.styles.css_styles import *


def custom_header():
    return HEADER_STILE + f"""<div class="custom-header"></div>"""


def custom_footer():
    html_content = (
    FOOTER_STYLE +
    "<div class=\"custom-footer\">" +
        "<p>Creadores:</p>" +
        "<a href=\"https://www.linkedin.com/in/pablo-oller-perez-7995721b2\" target=\"_blank\">Pablo Oller Pérez</a><br>" +
        "<a href=\"https://github.com/pabloquirce23\" target=\"_blank\">Pablo Santos Quirce</a><br>" +
        "<a href=\"https://github.com/acscr44\" target=\"_blank\">Alejandro Castillo Carmona</a>" +
    "</div>"
    )
    return html_content


def custom_title(title):
    return TITLE_STILE + f"""<div class="title"><h1 class='title'>{title}</h1></div><br>"""

def custom_width():
    return WIDTH_STILE

def description():
    html_content = f"""
    <div>
        <p><strong>Fraud Detect</strong> es una aplicación web diseñada para abordar de manera eficiente y precisa la detección de 
        posibles fraudes bancarios.<br>
        Su funcionalidad radica en la capacidad de procesar documentos en formato PDF, extrayendo las tablas contenidas en ellos 
        mediante su lector integrado. 
        A partir de los datos recopilados en estas tablas, la aplicación lleva a cabo un exhaustivo análisis para identificar 
        posibles irregularidades financieras que puedan indicar la presencia de actividades fraudulentas entre una lista de clientes.<br>
        Además muestra una sucesión de gráficas con datos que pueden ser de utilidad para el usuario.
        </p>
        <br><br>
    </div>
    """.strip().replace('\n', '')
    return html_content