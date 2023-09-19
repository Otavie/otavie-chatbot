class Chatbot {
    constructor(){
        this.args = {
            openButton: document.querySelector('.chatbot__button'),
            chatBot: document.querySelector('.chatbot__support'),
            sendButton: document.querySelector('.send__button')
        }
        this.state = false;
        this.messages = [];
    }
    display(){
        const {openButton, chatBot, sendButton} = this.args;
        openButton.addEventListener('click', ()=>this.toggleState(chatBot))
        sendButton.addEventListener('click', ()=>this.onSendButton(chatBot))
        const node = chatBot.querySelector('input');
        node.addEventListener("keyup", ({key})=>{
            if (key === "Enter"){
                this.onSendButton(chatBot);
            }
        })
    }
    toggleState(chatBot){
        this.state = !this.state;
        if (this.state){
            chatBot.classList.add('chatbot--active')
        } else {
            chatBot.classList.remove('chatbot--active');
        }
    }
    onSendButton(chatbot){
        var textField = chatbot.querySelector('input');
        let text1 = textField.value
        if (text1 === ""){
            return;
        }
        let message1 = { name: "User", message: text1 }
        this.messages.push(message1);
        fetch($SCRIPT_ROOT + '/predict', {
            method: 'POST',
            body: JSON.stringify({ message: text1 }),
            mode: 'cors',
            headers: {
                'Content-Type': 'application/json'
            },
        })
        .then(response => response.json())
        .then(response => {
            let message2 = { name: "Otavie", message: response.answer };
            this.messages.push(message2);
            this.updateChatText(chatbot);
            textField.value = '';

        }).catch((error) => {
            console.log('Error:', error);
            this.updateChatText(chatbot);
            textField.value = '';
        });
    }

    updateChatText(chatbot){
        var html = '';
        this.messages.slice().reverse().forEach(function(item, index){
            if (item.name === "Otavie")
            {
                html += '<div class="messages_item messages__item--visitor">' + item.message + '</div>';
            }
            else
            {
                html += '<div class="messages_item messages__item--operator">' + item.message + '</div>';
            }
        });
        const chatMessage = chatbot.querySelector('.chatbot__messages');
        chatMessage.innerHTML = html;
    }
}

const chatbot = new Chatbot();
chatbot.display()