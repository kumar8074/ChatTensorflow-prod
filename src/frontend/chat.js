// ===================================================================================
// Project: ChatTensorFlow
// File: src/frontend/chat.js
// Description: Frontend chat logic with streaming, thread management, and persistence
// Author: LALAN KUMAR
// ===================================================================================

class ChatManager {
    constructor() {
        this.API_BASE_URL = 'http://localhost:8000/api/rag';
        this.userId = this.getOrCreateUserId();
        this.currentThreadId = this.getCurrentThreadId();
        this.threads = this.loadThreads();
        this.currentMessages = [];
        
        this.initializeElements();
        this.attachEventListeners();
        this.loadCurrentThread();
        this.renderThreadList();
    }

    // ==================== User ID Management ====================
    getOrCreateUserId() {
        let userId = localStorage.getItem('tensorflow_user_id');
        if (!userId) {
            userId = `user_${this.generateId()}`;
            localStorage.setItem('tensorflow_user_id', userId);
        }
        return userId;
    }

    generateId() {
        return Date.now().toString(36) + Math.random().toString(36).substr(2);
    }

    // ==================== Thread Management ====================
    getCurrentThreadId() {
        const stored = localStorage.getItem('tensorflow_current_thread');
        return stored ? parseInt(stored) : 1;
    }

    setCurrentThreadId(threadId) {
        this.currentThreadId = threadId;
        localStorage.setItem('tensorflow_current_thread', threadId.toString());
    }

    createNewThread() {
        const maxThreadId = Math.max(0, ...Object.keys(this.threads).map(id => parseInt(id)));
        const newThreadId = maxThreadId + 1;
        
        this.setCurrentThreadId(newThreadId);
        this.currentMessages = [];
        this.clearChatArea();
        this.saveCurrentThread();
        this.renderThreadList();
    }

    // ==================== Thread Storage ====================
    loadThreads() {
        const stored = localStorage.getItem('tensorflow_threads');
        return stored ? JSON.parse(stored) : {};
    }

    saveThreads() {
        localStorage.setItem('tensorflow_threads', JSON.stringify(this.threads));
    }

    saveCurrentThread() {
        this.threads[this.currentThreadId] = {
            id: this.currentThreadId,
            messages: this.currentMessages,
            timestamp: Date.now(),
            title: this.generateThreadTitle()
        };
        this.saveThreads();
        this.renderThreadList();
    }

    loadCurrentThread() {
        const thread = this.threads[this.currentThreadId];
        if (thread && thread.messages) {
            this.currentMessages = thread.messages;
            this.renderMessages();
        }
    }

    switchThread(threadId) {
        this.saveCurrentThread();
        this.setCurrentThreadId(threadId);
        this.loadCurrentThread();
    }

    generateThreadTitle() {
        if (this.currentMessages.length === 0) {
            return 'New Conversation';
        }
        const firstUserMsg = this.currentMessages.find(m => m.role === 'user');
        if (firstUserMsg) {
            const title = firstUserMsg.content.substring(0, 30);
            return title.length < firstUserMsg.content.length ? title + '...' : title;
        }
        return `Thread ${this.currentThreadId}`;
    }

    // ==================== DOM Elements ====================
    initializeElements() {
        this.chatArea = document.querySelector('.chat-area');
        this.chatInput = document.querySelector('.chat-input');
        this.newChatBtn = document.querySelector('.new-chat-btn');
        this.threadContainer = document.querySelector('.sidebar');
    }

    attachEventListeners() {
        this.chatInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });

        this.newChatBtn.addEventListener('click', () => {
            this.createNewThread();
        });
    }

    // ==================== UI Rendering ====================
    clearChatArea() {
        this.chatArea.innerHTML = `
            <div class="welcome-logo"></div>
            <h1 class="welcome-text">What can I help with?</h1>
            <div class="input-container">
                <button class="attach-btn">+</button>
                <input type="text" class="chat-input" placeholder="Ask me anything about TensorFlow...">
            </div>
            <div class="model-selector">
                <select class="model-dropdown">
                    <option>Gemini-2.5-Flash (Default)</option>
                    <option>Claude Sonnet 4.5</option>
                    <option>GPT-4</option>
                </select>
            </div>
        `;
        this.chatArea.classList.remove('has-messages');
        this.initializeElements();
        this.attachEventListeners();
    }

    removeWelcomeScreen() {
        const welcome = this.chatArea.querySelector('.welcome-logo');
        const welcomeText = this.chatArea.querySelector('.welcome-text');
        const modelSelector = this.chatArea.querySelector('.model-selector');
        
        if (welcome) welcome.remove();
        if (welcomeText) welcomeText.remove();
        if (modelSelector) modelSelector.remove();
    }

    ensureMessagesContainer() {
        this.removeWelcomeScreen();
        if (!this.chatArea.classList.contains('has-messages')) {
            this.chatArea.classList.add('has-messages');
        }
        let container = this.chatArea.querySelector('.messages-container');
        if (!container) {
            container = document.createElement('div');
            container.className = 'messages-container';
            const inputContainer = this.chatArea.querySelector('.input-container');
            if (inputContainer) {
                this.chatArea.insertBefore(container, inputContainer);
            } else {
                this.chatArea.appendChild(container);
            }
        }
        return container;
    }

    renderMessages() {
        const container = this.ensureMessagesContainer();
        container.innerHTML = '';
        
        this.currentMessages.forEach(msg => {
            if (msg.role === 'user' || msg.role === 'assistant') {
                container.appendChild(this.createMessageElement(msg));
            }
        });
        
        this.scrollToBottom();
    }

    createMessageElement(message) {
        const msgDiv = document.createElement('div');
        msgDiv.className = `message ${message.role}-message`;
        
        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.textContent = message.role === 'user' ? '👤' : '🤖';
        
        const content = document.createElement('div');
        content.className = 'message-content';
        
        // Handle markdown-style code blocks with TensorFlow citations
        const formattedContent = this.formatContent(message.content);
        content.innerHTML = formattedContent;
        this.highlightCode(content);
        this.addCopyButtons(content);
        
        msgDiv.appendChild(avatar);
        msgDiv.appendChild(content);
        
        // Add node details if present
        if (message.nodeDetails && message.nodeDetails.length > 0) {
            const detailsToggle = this.createNodeDetailsToggle(message.nodeDetails);
            msgDiv.appendChild(detailsToggle);
        }
        
        return msgDiv;
    }

    highlightCode(element) {
        if (typeof Prism !== 'undefined') {
            Prism.highlightAllUnder(element);
        }
    }

    addCopyButtons(element) {
        const pres = element.querySelectorAll('pre');
        pres.forEach(pre => {
            if (pre.querySelector('.copy-btn')) return; // Avoid duplicates
            const button = document.createElement('button');
            button.className = 'copy-btn';
            button.innerHTML = '📋 Copy';
            button.addEventListener('click', async () => {
                const code = pre.querySelector('code');
                try {
                    await navigator.clipboard.writeText(code.textContent);
                    button.innerHTML = '✅ Copied!';
                    setTimeout(() => {
                        button.innerHTML = '📋 Copy';
                    }, 2000);
                } catch (err) {
                    console.error('Failed to copy', err);
                }
            });
            button.style.cssText = `
                position: absolute;
                top: 8px;
                right: 8px;
                background: rgba(255, 255, 255, 0.1);
                backdrop-filter: blur(10px);
                border: 1px solid rgba(255, 255, 255, 0.1);
                color: var(--foreground);
                padding: 6px 12px;
                border-radius: var(--radius-sm);
                cursor: pointer;
                font-size: 11px;
                font-family: var(--font-sans);
                transition: all 0.2s;
                z-index: 10;
                opacity: 0;
            `;
            
            pre.addEventListener('mouseenter', () => {
                button.style.opacity = '1';
            });
            
            pre.addEventListener('mouseleave', () => {
                if (!button.innerHTML.includes('✅')) {
                    button.style.opacity = '0';
                }
            });
            
            button.addEventListener('mouseenter', () => {
                button.style.background = 'rgba(188, 177, 253, 0.2)';
                button.style.borderColor = 'var(--primary)';
            });
            button.addEventListener('mouseleave', () => {
                button.style.background = 'rgba(255, 255, 255, 0.1)';
                button.style.borderColor = 'rgba(255, 255, 255, 0.1)';
            });
            pre.style.position = 'relative';
            pre.appendChild(button);
        });
    }

    formatContent(content = '') {
        if (!content) return '';

    
        const citations = [];
        content = content.replace(/\[URL:\s*(https?:\/\/[^\]]+)\]/g, (match, url) => {
            const index = citations.length;
            citations.push(url);
            return `__CITATION_${index}__`;
        });

    
        content = content.replace(/```(\w+)?\n([\s\S]*?)```/g, (match, lang, code) => {
            const language = lang || 'python';
            return `<pre><code class="language-${language}">${this.escapeHtml(code.trim())}</code></pre>`;
        });

    
        const parts = content.split(/(<pre[\s\S]*?<\/pre>)/g);

   
        for (let i = 0; i < parts.length; i++) {
            const part = parts[i];

            // Skip <pre> blocks — do NOT modify them
            if (part.startsWith('<pre')) continue;

            let processed = part;

        
            processed = processed.replace(/^### (.+)$/gm, '<h3>$1</h3>');
            processed = processed.replace(/^## (.+)$/gm, '<h2>$1</h2>');
            processed = processed.replace(/^# (.+)$/gm, '<h1>$1</h1>');

        
            processed = processed.replace(/\*\*([^\*]+?)\*\*/g, '<strong>$1</strong>');
            processed = processed.replace(/(?<!\*)\*(?!\*)([^\*]+?)\*(?!\*)/g, '<em>$1</em>');

        
            processed = processed.replace(/`([^`]+)`/g, '<code>$1</code>');

        
            processed = processed.replace(/__CITATION_(\d+)__/g, (match, index) => {
                return this.formatCitation(citations[parseInt(index)]);
            });

        
            processed = processed.replace(/(https?:\/\/[^\s<]+)/g, (match, offset, full) => {
                const before = full.slice(offset - 15, offset);
                const insideAttribute = before.includes('href="') || before.includes("href='");

                if (insideAttribute) return match; // skip existing <a> tags

                return this.formatUrl(match);
            });

            processed = processed.replace(/\n/g, '<br>');

            // Save
            parts[i] = processed;
        }

        return parts.join('');
    }


    formatCitation(url) {
        try {
            const urlObj = new URL(url);
            let label = urlObj.pathname.split('/').filter(p => p).pop() || '';
            
            // Remove file extensions
            label = label.replace(/\.(html|php|aspx)$/i, '');
            
            // Handle anchor links
            if (urlObj.hash) {
                const anchor = urlObj.hash.slice(1);
                label = decodeURIComponent(anchor)
                    .replace(/[_-]/g, ' ')
                    .replace(/%[0-9A-F]{2}/gi, ' ');
            }
            
            // Clean up the label
            label = label.replace(/[_-]/g, ' ').trim();
            
            // Fallback to domain if empty
            if (!label || label.length < 2) {
                label = urlObj.hostname.replace('www.', '');
            }
            
            // Truncate if too long
            if (label.length > 40) {
                label = label.substring(0, 37) + '...';
            }
            
            return `<a href="${url}" target="_blank" class="citation-link" title="${url}">[${label}]</a>`;
        } catch {
            return `<a href="${url}" target="_blank" class="citation-link">[source]</a>`;
        }
    }

    formatUrl(url) {
        try {
            const urlObj = new URL(url);
            const label = urlObj.hostname.replace('www.', '');
            return `<a href="${url}" target="_blank" class="external-link">${label}</a>`;
        } catch {
            return `<a href="${url}" target="_blank" class="external-link">${url}</a>`;
        }
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    createNodeDetailsToggle(nodeDetails) {
        const container = document.createElement('div');
        container.className = 'node-details-container';
        
        const toggle = document.createElement('button');
        toggle.className = 'node-details-toggle';
        toggle.innerHTML = '▼ Show processing steps';
        toggle.dataset.expanded = 'false';
        
        const detailsContent = document.createElement('div');
        detailsContent.className = 'node-details-content hidden';
        
        nodeDetails.forEach(detail => {
            const step = document.createElement('div');
            step.className = 'node-step';
            step.innerHTML = `
                <div class="node-step-header">
                    <span class="node-name">${this.formatNodeName(detail.node)}</span>
                    <span class="node-time">${new Date(detail.timestamp).toLocaleTimeString()}</span>
                </div>
                ${detail.data ? `<div class="node-data">${this.escapeHtml(JSON.stringify(detail.data, null, 2))}</div>` : ''}
            `;
            detailsContent.appendChild(step);
        });
        
        toggle.addEventListener('click', () => {
            const isExpanded = toggle.dataset.expanded === 'true';
            toggle.dataset.expanded = !isExpanded;
            toggle.innerHTML = isExpanded ? '▼ Show processing steps' : '▲ Hide processing steps';
            detailsContent.classList.toggle('hidden');
        });
        
        container.appendChild(toggle);
        container.appendChild(detailsContent);
        
        return container;
    }

    scrollToBottom() {
        const container = this.chatArea.querySelector('.messages-container');
        if (container) {
            container.scrollTop = container.scrollHeight;
        }
    }

    // ==================== Message Sending ====================
    async sendMessage() {
        const message = this.chatInput.value.trim();
        if (!message) return;
        
        // Add user message to UI
        this.currentMessages.push({
            role: 'user',
            content: message,
            timestamp: Date.now()
        });
        
        this.renderMessages();
        this.chatInput.value = '';
        
        // Create assistant message placeholder
        const assistantMsg = {
            role: 'assistant',
            content: '',
            timestamp: Date.now(),
            nodeDetails: []
        };
        this.currentMessages.push(assistantMsg);
        
        // Start streaming
        await this.streamResponse(message, assistantMsg);
        
        this.saveCurrentThread();
    }

    async streamResponse(message, assistantMsg) {
        const container = this.ensureMessagesContainer();
        const msgElement = this.createMessageElement(assistantMsg);
        container.appendChild(msgElement);
    
        const contentDiv = msgElement.querySelector('.message-content');
        const nodeDetails = [];
    
        try {
            const response = await fetch(`${this.API_BASE_URL}/ask/stream`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    user_id: this.userId,
                    message: message,
                    thread_id: `${this.userId}_${this.currentThreadId}`
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';
            let fullResponse = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });

                const lines = buffer.split('\n');
                buffer = lines.pop();

                for (const line of lines) {
                    if (!line.startsWith('data: ')) continue;

                    const jsonStr = line.slice(6).trim();
                    if (!jsonStr) continue;

                    try {
                        const parsed = JSON.parse(jsonStr);

                        // Get the event type (use 'type' field from StreamEvent schema)
                        const eventType = parsed.type || parsed.event_type || 'update';

                        // STREAMING PROGRESSION
                        if (eventType === 'node' || eventType === 'update') {
                            // Track node updates
                            nodeDetails.push({
                                node: parsed.node,
                                data: parsed.data,
                                timestamp: Date.now()
                            });

                            this.showNodeProgress(msgElement, parsed.node);

                            // Standard message streaming
                            if (parsed.data && parsed.data.messages) {
                                const lastMsg = parsed.data.messages.at(-1);
                                if (lastMsg && lastMsg.content) {
                                    fullResponse = lastMsg.content;
                                    contentDiv.innerHTML = this.formatContent(fullResponse);
                                    this.highlightCode(contentDiv);
                                    this.addCopyButtons(contentDiv);
                                    this.scrollToBottom();
                                }
                            }
                        }
                        // Handle response_chunk type
                        else if (eventType === 'response_chunk') {
                            // Track node updates
                            nodeDetails.push({
                                node: parsed.node,
                                data: parsed.data,
                                timestamp: Date.now()
                            });

                            this.showNodeProgress(msgElement, parsed.node);

                            // Response chunk with direct content
                            if (parsed.data && parsed.data.messages) {
                                const lastMsg = parsed.data.messages.at(-1);
                                if (lastMsg && lastMsg.content) {
                                    fullResponse = lastMsg.content;
                                    contentDiv.innerHTML = this.formatContent(fullResponse);
                                    this.highlightCode(contentDiv);
                                    this.addCopyButtons(contentDiv);
                                    this.scrollToBottom();
                                }
                            }
                        }

                        // END OF STREAM
                        else if (eventType === 'end') {
                            this.hideNodeProgress(msgElement);

                            assistantMsg.content = fullResponse;

                            contentDiv.innerHTML = this.formatContent(fullResponse);
                            this.highlightCode(contentDiv);
                            this.addCopyButtons(contentDiv);

                            if (nodeDetails.length > 0) {
                                assistantMsg.nodeDetails = nodeDetails;
                                const toggle = this.createNodeDetailsToggle(nodeDetails);
                                msgElement.appendChild(toggle);
                            }
                        }
                        // ERROR
                        else if (eventType === 'error') {
                            this.hideNodeProgress(msgElement);
                            const errorMsg = parsed.data?.error || 'An error occurred';
                            contentDiv.innerHTML = `<span class="error">Error: ${errorMsg}</span>`;
                        }

                    } catch (err) {
                        console.error('Error parsing SSE chunk:', err);
                    }
                }
            }

            assistantMsg.content = fullResponse;

        } catch (error) {
            console.error('Streaming error:', error);
            contentDiv.innerHTML = `<span class="error">Error: ${error.message}</span>`;
        }
    }

    showNodeProgress(msgElement, nodeName) {
        let progress = msgElement.querySelector('.node-progress');
        if (!progress) {
            progress = document.createElement('div');
            progress.className = 'node-progress';
            msgElement.querySelector('.message-content').appendChild(progress);
        }
        progress.innerHTML = `<span class="loading-spinner"></span> ${this.formatNodeName(nodeName)}...`;
    }

    hideNodeProgress(msgElement) {
        const progress = msgElement.querySelector('.node-progress');
        if (progress) {
            progress.remove();
        }
    }

    formatNodeName(nodeName) {
        // Map specific node names to user-friendly labels
        const nameMap = {
            'analyze_and_route_query': 'Analyzing Query',
            'create_research_plan': 'Creating Research Plan',
            'conduct_research': 'Researching',
            'respond': 'Generating Response',
            'summarize_conversation': 'Summarizing',
            'ask_for_more_info': 'Processing',
            'respond_to_general_query': 'Processing'
        };
        
        return nameMap[nodeName] || (nodeName || 'processing')
            .split('_')
            .map(word => word.charAt(0).toUpperCase() + word.slice(1))
            .join(' ');
    }

    // ==================== Thread List Rendering ====================
    renderThreadList() {
        const threadSections = this.groupThreadsByTime();
        
        // Find or create thread sections
        let yesterdaySection = this.threadContainer.querySelector('.thread-section:nth-child(3)');
        let previousSection = this.threadContainer.querySelector('.thread-section:nth-child(4)');
        
        // Clear existing threads (but keep section structure)
        if (yesterdaySection) {
            const title = yesterdaySection.querySelector('.section-title');
            yesterdaySection.innerHTML = '';
            if (title) yesterdaySection.appendChild(title);
        }
        
        if (previousSection) {
            const title = previousSection.querySelector('.section-title');
            previousSection.innerHTML = '';
            if (title) previousSection.appendChild(title);
        }
        
        // Render yesterday's threads
        if (threadSections.yesterday.length > 0) {
            threadSections.yesterday.forEach(thread => {
                const threadElement = this.createThreadElement(thread);
                yesterdaySection.appendChild(threadElement);
            });
        }
        
        // Render previous 7 days threads
        if (threadSections.previous7Days.length > 0) {
            threadSections.previous7Days.forEach(thread => {
                const threadElement = this.createThreadElement(thread);
                previousSection.appendChild(threadElement);
            });
        }
    }

    groupThreadsByTime() {
        const now = Date.now();
        const oneDayMs = 24 * 60 * 60 * 1000;
        const sevenDaysMs = 7 * oneDayMs;
        
        const yesterday = [];
        const previous7Days = [];
        
        Object.values(this.threads).forEach(thread => {
            const age = now - thread.timestamp;
            
            if (age < oneDayMs) {
                yesterday.push(thread);
            } else if (age < sevenDaysMs) {
                previous7Days.push(thread);
            }
        });
        
        // Sort by timestamp (newest first)
        yesterday.sort((a, b) => b.timestamp - a.timestamp);
        previous7Days.sort((a, b) => b.timestamp - a.timestamp);
        
        return { yesterday, previous7Days };
    }

    createThreadElement(thread) {
        const threadDiv = document.createElement('div');
        threadDiv.className = 'thread-item';
        if (thread.id === this.currentThreadId) {
            threadDiv.classList.add('active');
        }
        
        const threadName = document.createElement('div');
        threadName.className = 'thread-name';
        threadName.textContent = thread.title;
        
        const threadTime = document.createElement('div');
        threadTime.className = 'thread-time';
        threadTime.textContent = this.getRelativeTime(thread.timestamp);
        
        threadDiv.appendChild(threadName);
        threadDiv.appendChild(threadTime);
        
        threadDiv.addEventListener('click', () => {
            this.switchThread(thread.id);
            this.updateActiveThread();
        });
        
        return threadDiv;
    }

    updateActiveThread() {
        const allThreads = this.threadContainer.querySelectorAll('.thread-item');
        allThreads.forEach(t => t.classList.remove('active'));
        
        const activeThread = Array.from(allThreads).find(t => 
            t.querySelector('.thread-name').textContent === this.threads[this.currentThreadId]?.title
        );
        
        if (activeThread) {
            activeThread.classList.add('active');
        }
    }

    getRelativeTime(timestamp) {
        const now = Date.now();
        const diff = now - timestamp;
        const minutes = Math.floor(diff / 60000);
        const hours = Math.floor(diff / 3600000);
        const days = Math.floor(diff / 86400000);
        
        if (minutes < 1) return 'Just now';
        if (minutes < 60) return `${minutes}m ago`;
        if (hours < 24) return `${hours}h ago`;
        if (days === 1) return 'Yesterday';
        return `${days} days ago`;
    }
}

// Initialize chat manager when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.chatManager = new ChatManager();
    if (typeof Prism !== 'undefined') {
        Prism.highlightAll();
    }
});

// Add CSS for new elements dynamically
const style = document.createElement('style');
style.textContent = `
    .messages-container {
        flex: 1;
        overflow-y: auto;
        padding: calc(var(--spacing) * 6) 0;
        margin-bottom: calc(var(--spacing) * 4);
    }

    .message {
        display: flex;
        gap: calc(var(--spacing) * 4);
        margin-bottom: calc(var(--spacing) * 8);
        animation: fadeIn 0.3s ease-in;
        padding: 0 calc(var(--spacing) * 6);
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .message-avatar {
        width: calc(var(--spacing) * 10);
        height: calc(var(--spacing) * 10);
        border-radius: var(--radius-lg);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 24px;
        flex-shrink: 0;
        background: var(--muted);
    }

    .user-message .message-avatar {
        background: linear-gradient(135deg, #4a9eff, #357abd);
    }

    .assistant-message .message-avatar {
        background: linear-gradient(135deg, var(--primary), #9b8aff);
    }

    .message-content {
        flex: 1;
        color: var(--foreground);
        line-height: 1.7;
        font-size: 15px;
        max-width: 100%;
    }

    .message-content h1, .message-content h2, .message-content h3 {
        margin: calc(var(--spacing) * 6) 0 calc(var(--spacing) * 3) 0;
        color: var(--foreground);
        font-weight: var(--font-weight-bold);
    }

    .message-content h1 {
        font-size: 24px;
        border-bottom: 2px solid var(--border);
        padding-bottom: calc(var(--spacing) * 2);
    }

    .message-content h2 {
        font-size: 20px;
    }

    .message-content h3 {
        font-size: 17px;
        color: var(--primary);
    }

    .message-content strong {
        font-weight: var(--font-weight-bold);
        color: var(--foreground);
    }

    .message-content em {
        font-style: italic;
        color: var(--muted-foreground);
    }

    .message-content code {
        background: var(--muted);
        padding: calc(var(--spacing) * 0.5) calc(var(--spacing) * 2);
        border-radius: var(--radius-xs);
        font-family: 'Monaco', 'Menlo', 'Consolas', monospace;
        font-size: 13px;
        color: var(--primary);
    }

    .message-content pre {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: var(--radius-lg);
        padding: calc(var(--spacing) * 4);
        overflow-x: auto;
        margin: calc(var(--spacing) * 4) 0;
        box-shadow: 0 2px 8px oklch(from var(--border) l c h / 0.3);
        position: relative;
        max-width: 100%;
    }

    .message-content pre code {
        background: none;
        padding: 0;
        color: var(--foreground);
        font-size: 13px;
        line-height: 1.5;
        display: block;
    }

    .citation-link {
        display: inline-flex;
        align-items: center;
        color: var(--primary);
        text-decoration: none;
        font-size: 13px;
        padding: 2px 6px;
        border-radius: var(--radius-xs);
        background: oklch(from var(--primary) l c h / 0.1);
        border: 1px solid oklch(from var(--primary) l c h / 0.2);
        margin: 0 2px;
        transition: all 0.2s;
        white-space: nowrap;
    }

    .citation-link:hover {
        background: oklch(from var(--primary) l c h / 0.2);
        border-color: var(--primary);
        transform: translateY(-1px);
    }

    .external-link {
        color: var(--primary);
        text-decoration: none;
        border-bottom: 1px solid var(--primary);
        padding-bottom: 1px;
        transition: opacity 0.2s;
    }

    .external-link:hover {
        opacity: 0.7;
    }

    .node-progress {
        display: flex;
        align-items: center;
        gap: calc(var(--spacing) * 2);
        margin-top: calc(var(--spacing) * 3);
        padding: calc(var(--spacing) * 2) calc(var(--spacing) * 3);
        background: var(--muted);
        border-radius: var(--radius-md);
        font-size: 13px;
        color: var(--muted-foreground);
    }

    .loading-spinner {
        width: 12px;
        height: 12px;
        border: 2px solid var(--border);
        border-top-color: var(--primary);
        border-radius: 50%;
        animation: spin 0.8s linear infinite;
    }

    @keyframes spin {
        to { transform: rotate(360deg); }
    }

    .node-details-container {
        margin-top: calc(var(--spacing) * 4);
        margin-left: calc(var(--spacing) * 14);
    }

    .node-details-toggle {
        background: var(--muted);
        border: none;
        color: var(--muted-foreground);
        padding: calc(var(--spacing) * 2) calc(var(--spacing) * 3);
        border-radius: var(--radius-md);
        font-size: 12px;
        cursor: pointer;
        transition: all 0.2s;
        display: flex;
        align-items: center;
        gap: calc(var(--spacing) * 2);
    }

    .node-details-toggle:hover {
        background: var(--sidebar-accent);
        color: var(--foreground);
    }

    .node-details-content {
        margin-top: calc(var(--spacing) * 3);
        padding: calc(var(--spacing) * 4);
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: var(--radius-lg);
        max-height: 400px;
        overflow-y: auto;
    }

    .node-details-content.hidden {
        display: none;
    }

    .node-step {
        margin-bottom: calc(var(--spacing) * 4);
        padding-bottom: calc(var(--spacing) * 4);
        border-bottom: 1px solid var(--border);
    }

    .node-step:last-child {
        border-bottom: none;
        margin-bottom: 0;
        padding-bottom: 0;
    }

    .node-step-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: calc(var(--spacing) * 2);
    }

    .node-name {
        color: var(--primary);
        font-weight: var(--font-weight-medium);
        font-size: 13px;
    }

    .node-time {
        color: var(--muted-foreground);
        font-size: 11px;
    }

    .node-data {
        background: var(--background);
        padding: calc(var(--spacing) * 3);
        border-radius: var(--radius-sm);
        font-family: 'Monaco', 'Menlo', monospace;
        font-size: 12px;
        color: var(--muted-foreground);
        white-space: pre-wrap;
        word-break: break-word;
        max-height: 200px;
        overflow-y: auto;
    }

    .thread-item.active {
        background: var(--sidebar-accent);
        border-left: 3px solid var(--primary);
    }

    .error {
        color: #ff6b6b;
        background: rgba(255, 107, 107, 0.1);
        padding: calc(var(--spacing) * 2);
        border-radius: var(--radius-md);
        border-left: 3px solid #ff6b6b;
    }

    .chat-area {
        position: relative;
        display: flex;
        flex-direction: column;
    }

    .input-container {
        flex-shrink: 0;
        width: 100%;
        max-width: 800px;
        position: relative;
        padding: calc(var(--spacing) * 4) 0;
        background: var(--background);
    }

    .chat-area.has-messages .input-container {
        padding: 0;
        background: transparent;
        position: sticky;
        bottom: 0;
        padding: calc(var(--spacing) * 4) calc(var(--spacing) * 6);
        background: linear-gradient(to top, var(--background) 80%, transparent);
    }

    /* Better list styling */
    .message-content ol {
        margin: calc(var(--spacing) * 3) 0;
        padding-left: calc(var(--spacing) * 6);
    }

    .message-content ol li {
        margin-bottom: calc(var(--spacing) * 2);
        line-height: 1.6;
    }

    .message-content ul {
        margin: calc(var(--spacing) * 3) 0;
        padding-left: calc(var(--spacing) * 6);
        list-style-type: disc;
    }

    .message-content ul li {
        margin-bottom: calc(var(--spacing) * 2);
        line-height: 1.6;
    }

    /* Paragraph spacing */
    .message-content p {
        margin-bottom: calc(var(--spacing) * 3);
    }

    /* Scrollbar styling for code blocks */
    .message-content pre::-webkit-scrollbar {
        height: 8px;
    }

    .message-content pre::-webkit-scrollbar-track {
        background: var(--muted);
        border-radius: var(--radius-sm);
    }

    .message-content pre::-webkit-scrollbar-thumb {
        background: var(--border);
        border-radius: var(--radius-sm);
    }

    .message-content pre::-webkit-scrollbar-thumb:hover {
        background: var(--muted-foreground);
    }

    /* Better message spacing on mobile */
    @media (max-width: 768px) {
        .message {
            padding: 0 calc(var(--spacing) * 3);
        }

        .message-content {
            font-size: 14px;
        }

        .message-avatar {
            width: calc(var(--spacing) * 8);
            height: calc(var(--spacing) * 8);
            font-size: 20px;
        }

        .node-details-container {
            margin-left: calc(var(--spacing) * 12);
        }
    }
`;
document.head.appendChild(style);