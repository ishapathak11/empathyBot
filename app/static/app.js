// app/static/app.js
document.addEventListener('DOMContentLoaded', function() {
    // Generate a session ID or get from localStorage
    let sessionId = localStorage.getItem('chatSessionId');
    if (!sessionId) {
        sessionId = generateUUID();
        localStorage.setItem('chatSessionId', sessionId);
    }
    
    // Elements
    const messagesArea = document.getElementById('messages');
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');
    const optionsArea = document.getElementById('options-area');
    const emojiArea = document.getElementById('emoji-area');
    const exitBtn = document.getElementById('exit-btn');
    const endChatModal = document.getElementById('end-chat-modal');
    const confirmEndBtn = document.getElementById('confirm-end-btn');
    const cancelEndBtn = document.getElementById('cancel-end-btn');
    
    let chatActive = true;
    let socket;
    
    // Define updated emotion mapping
    const emotionClassMap = {
        'joy': 'joy',
        'sadness': 'sadness',
        'fear': 'fear',
        'anger': 'anger',
        'neutral': 'neutral',
        'love': 'love',         // New class
        'gratitude': 'gratitude', // New class
        'curiosity': 'curiosity', // New class
        'disappointment': 'disappointment', // New class
        'surprise': 'surprise', // New class
        'pride': 'pride'        // New class
    };
    
    // Connect to WebSocket
    function connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/${sessionId}`;
        
        socket = new WebSocket(wsUrl);
        
        socket.onopen = function(e) {
            console.log('WebSocket connection established');
            chatActive = true;
        };
        
        socket.onmessage = function(event) {
            const data = JSON.parse(event.data);
            
            // Handle ping messages
            if (data.type === 'ping') {
                socket.send(JSON.stringify({type: 'pong'}));
                return;
            }
            
            // Normal message handling
            handleIncomingMessage(data);
        };
        
        socket.onclose = function(event) {
            console.log('WebSocket connection closed');
            // Try to reconnect if not explicitly ended
            if (chatActive) {
                setTimeout(connectWebSocket, 3000);
            }
        };
        
        socket.onerror = function(error) {
            console.error('WebSocket error:', error);
        };
    }
    
    // Handle incoming messages
    function handleIncomingMessage(data) {
        if (data.type === 'message') {
            // Skip duplicate welcome messages
            if (data.content === "Hi there! I'm EmpathyBot, here to provide a bit of positivity when you need it." &&
                document.querySelectorAll('.bot-message').length > 0) {
                console.log("Skipping duplicate welcome message");
                return;
            }
            
            // Add message to UI
            addMessage(data.role, data.content, data.emotion);
            
            // Show emoji area for bot messages if chat is active
            if (data.role === 'bot' && chatActive) {
                emojiArea.style.display = 'flex';
                emojiArea.dataset.messageId = document.querySelectorAll('.bot-message').length - 1;
            }
            
            // Check if this is a goodbye message
            if (data.is_goodbye) {
                endChat(false); // End chat without sending another message
            }
        } else if (data.type === 'chat_ended') {
            endChat(false);
        } else if (data.type === 'acknowledgment') {
            console.log(data.text);
        }
    }
    
    // Send message
    function sendMessage() {
        if (!chatActive) return;
        
        const message = userInput.value.trim();
        if (!message) return;
        
        // Add user message to UI
        addMessage('user', message);
        
        // Send to server
        socket.send(JSON.stringify({
            type: 'message',
            content: message
        }));
        
        // Clear input
        userInput.value = '';
        
        // Hide emoji area
        emojiArea.style.display = 'none';
    }
    
    // Add message to UI
    function addMessage(role, content, emotion) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message');
        
        // Clean up any formatting placeholders in the content
        content = content.replace(/_comma_/g, ',');
        content = content.replace(/no_comma_/g, '');
        
        if (role === 'bot') {
            messageDiv.classList.add('bot-message');
            
            // Create emotion badge if emotion is detected
            if (emotion) {
                // Map emotion to CSS class
                const cssClass = emotionClassMap[emotion.toLowerCase()] || 'neutral';
                
                const emotionBadge = document.createElement('div');
                emotionBadge.classList.add('emotion-badge');
                emotionBadge.classList.add(`emotion-${cssClass}`);
                
                // Capitalize first letter for display
                const displayText = emotion.charAt(0).toUpperCase() + emotion.slice(1);
                emotionBadge.textContent = displayText;
                
                messageDiv.appendChild(emotionBadge);
            }
            
            // Create message content
            const contentSpan = document.createElement('div');
            contentSpan.classList.add('message-content');
            contentSpan.textContent = content;
            messageDiv.appendChild(contentSpan);
        } else {
            messageDiv.classList.add('user-message');
            messageDiv.textContent = content;
        }
        
        messagesArea.appendChild(messageDiv);
        
        // Scroll to bottom
        messagesArea.scrollTop = messagesArea.scrollHeight;
    }
    
    // Add chat ended notification
    function addChatEndedNotification() {
        const endDiv = document.createElement('div');
        endDiv.classList.add('chat-ended');
        endDiv.textContent = 'Chat session ended. Refresh the page to start a new chat.';
        messagesArea.appendChild(endDiv);
        messagesArea.scrollTop = messagesArea.scrollHeight;
    }
    
    // End chat function
    function endChat(sendEndSignal = true) {
        if (!chatActive) return;
        
        chatActive = false;
        
        // Disable input
        userInput.disabled = true;
        sendBtn.disabled = true;
        userInput.placeholder = "Chat ended";
        
        // Hide emoji area
        emojiArea.style.display = 'none';
        
        // Add notification
        addChatEndedNotification();
        
        // Send end signal to server
        if (sendEndSignal && socket.readyState === WebSocket.OPEN) {
            socket.send(JSON.stringify({
                type: 'end_chat'
            }));
        }
    }
    
    // Show exit confirmation modal
    function showExitModal() {
        endChatModal.style.display = 'flex';
    }
    
    // Hide exit confirmation modal
    function hideExitModal() {
        endChatModal.style.display = 'none';
    }
    
    // Generate UUID for session ID
    function generateUUID() {
        return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
            const r = Math.random() * 16 | 0;
            const v = c === 'x' ? r : (r & 0x3 | 0x8);
            return v.toString(16);
        });
    }
    
    // Event listeners
    sendBtn.addEventListener('click', sendMessage);
    
    userInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });
    
    // Exit button
    exitBtn.addEventListener('click', showExitModal);
    
    // Confirm end chat
    confirmEndBtn.addEventListener('click', function() {
        hideExitModal();
        endChat(true);
    });
    
    // Cancel end chat
    cancelEndBtn.addEventListener('click', hideExitModal);
    
    // Emoji feedback
    document.querySelectorAll('.emoji-btn').forEach(btn => {
        btn.addEventListener('click', function() {
            if (!chatActive) return;
            
            const messageId = parseInt(emojiArea.dataset.messageId);
            const emoji = this.dataset.emoji;
            
            socket.send(JSON.stringify({
                type: 'feedback',
                message_id: messageId,
                feedback: emoji
            }));
            
            emojiArea.style.display = 'none';
        });
    });
    
    // Initialize WebSocket connection
    connectWebSocket();
});