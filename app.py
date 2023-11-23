import numpy as np
from flask import Flask, jsonify, request
from PIL import Image

import skin_cancer_detection as SCD

app = Flask(__name__)

@app.route("/api/runmodel", methods=["POST"])
def run_model():
    try:
        pic = request.files["pic"]
        inputimg = Image.open(pic)
        inputimg = inputimg.resize((28, 28))
        img = np.array(inputimg).reshape(-1, 28, 28, 3)
        result = SCD.model.predict(img)

        result = result.tolist()
        # print(result)
        max_prob = max(result[0])
        class_ind = result[0].index(max_prob)
        # print(class_ind)
        result_class = SCD.classes[class_ind]

        if class_ind == 0:
            info = (
                "A ceratose actínica, também chamada de ceratose solar ou senil, "
                "refere-se a uma condição em que as células da camada superior da pele apresentam mudanças anormais. "
                "Isso é considerado uma lesão que pode se tornar cancerígena ou até mesmo já estar no estágio inicial de "
                "um tipo de câncer de pele chamado carcinomas de células escamosas. "
                "Resumindo, é uma condição séria que precisa de atenção devido ao risco de desenvolver câncer de pele."
            )
        elif class_ind == 1:
            info = (
                "O carcinoma de células basais é um tipo de câncer de pele. "
                "Ele tem origem nas células basais, que são um tipo de célula na pele responsável "
                "por produzir novas células de pele à medida que as antigas morrem. "
                "Geralmente, o carcinoma de células basais se manifesta como um pequeno caroço "
                "levemente transparente na pele, embora possa assumir outras formas. "
                "Este tipo de câncer ocorre com maior frequência em áreas da pele expostas ao sol, "
                "como a cabeça e o pescoço."
            )
        elif class_ind == 2:
            info = (
                "A ceratose liquenoide benigna (BLK) geralmente se apresenta como uma lesão solitária "
                "que ocorre predominantemente no tronco e nos membros superiores em mulheres de meia-idade. "
                "A patogênese do BLK não é clara; no entanto, sugere-se que o BLK pode estar associado à "
                "fase inflamatória do lentigo solar (SL) em regressão."
            )
        elif class_ind == 3:
            info = (
                "Os dermatofibromas são pequenos crescimentos cutâneos não cancerígenos (benignos) "
                "que podem se desenvolver em qualquer parte do corpo, mas aparecem com mais frequência "
                "nas pernas inferiores, braços superiores ou parte superior das costas. Esses nódulos são comuns "
                "em adultos, mas são raros em crianças. Eles podem ter coloração rosa, cinza, vermelha ou marrom "
                "e podem mudar de cor ao longo dos anos. São firmes e muitas vezes têm uma sensação semelhante a uma pedra sob a pele."
            )
        elif class_ind == 4:
            info = (
                "Um nevo melanocítico (também conhecido como nevo citocítico, nevo de células e comumente como uma pinta) "
                "é um tipo de tumor melanocítico que contém células de nevo. Algumas fontes igualam o termo pinta com 'nevo melanocítico', "
                "mas há também fontes que equiparam o termo pinta com qualquer forma de nevo."
            )
        elif class_ind == 5:
            info = (
                "Os granulomas piogênicos são crescimentos cutâneos que são pequenos, redondos "
                "e geralmente de cor vermelho sangue. Tendem a sangrar porque contêm um grande número de vasos sanguíneos. "
                "São também conhecidos como hemangioma capilar lobular ou granuloma telangiectático."
            )
        elif class_ind == 6:
            info = (
                "O melanoma, o tipo mais sério de câncer de pele, desenvolve-se nas células (melanócitos) "
                "que produzem melanina - o pigmento que dá cor à sua pele. O melanoma também pode se formar nos olhos "
                "e, raramente, dentro do seu corpo, como no nariz ou na garganta. A causa exata de todos os melanomas não é clara, "
                "mas a exposição à radiação ultravioleta (UV) do sol ou de lâmpadas e camas de bronzeamento aumenta o risco de desenvolver melanoma."
            )
        response_data = {
            "result": result_class,
            "info": info
        }
        return jsonify(response_data)
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)