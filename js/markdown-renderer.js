/**
 * markdown-renderer.js
 * 动态渲染Markdown文档的核心功能
 */

class MarkdownRenderer {
    constructor() {
        this.config = null;
        this.currentDoc = 'main';
        this.baseUrl = 'docs/';
        this.init();
    }

    async init() {
        await this.loadConfig();
        await this.loadMarkdown(this.currentDoc);
        this.setupSidebar();
        this.setupSearch();
    }

    async loadConfig() {
        try {
            const response = await fetch(`${this.baseUrl}config.json`);
            this.config = await response.json();
        } catch (error) {
            console.error('Failed to load docs config:', error);
            this.config = { sections: [], sidebar: [] };
        }
    }

    async loadMarkdown(docName = 'main') {
        try {
            const response = await fetch(`${this.baseUrl}${docName}.md`);
            if (!response.ok) {
                throw new Error(`Failed to load ${docName}.md: ${response.status}`);
            }
            const markdown = await response.text();
            const html = this.parseMarkdown(markdown);
            this.renderContent(html);
            this.currentDoc = docName;
            this.updateSidebar(docName);
            this.addCodeHighlighting();
            this.addCopyButtons();
            this.updatePageTitle(docName);
        } catch (error) {
            console.error('Failed to load markdown:', error);
            this.renderError(docName);
        }
    }

    parseMarkdown(markdown) {
        // 简单的Markdown解析器
        let html = markdown;

        // Headers with anchor links
        html = html.replace(/^### (.*$)/gim, '<h3 id="$1">$1</h3>');
        html = html.replace(/^## (.*$)/gim, '<h2 id="$1">$1</h2>');
        html = html.replace(/^# (.*$)/gim, '<h1 id="$1">$1</h1>');

        // Code blocks
        html = html.replace(/```(\w+)?\n([\s\S]*?)```/g, (match, lang, code) => {
            const language = lang || 'text';
            return `<div class="code-block">
                <div class="code-header">
                    <span class="code-language">${language}</span>
                    <button class="copy-btn" onclick="copyCode(this)">Copy</button>
                </div>
                <pre><code class="language-${language}">${this.escapeHtml(code.trim())}</code></pre>
            </div>`;
        });

        // Inline code
        html = html.replace(/`([^`]+)`/g, '<code class="inline-code">$1</code>');

        // Bold
        html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

        // Italic
        html = html.replace(/\*(.*?)\*/g, '<em>$1</em>');

        // Links
        html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank">$1</a>');

        // Tables
        html = html.replace(/\|(.+)\|/g, (match, content) => {
            const cells = content.split('|').map(cell => cell.trim());
            return '<tr>' + cells.map(cell => `<td>${cell}</td>`).join('') + '</tr>';
        });

        // Lists
        html = html.replace(/^\* (.*$)/gim, '<li>$1</li>');
        html = html.replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>');
        html = html.replace(/^\d+\. (.*$)/gim, '<li>$1</li>');

        // Paragraphs
        html = html.replace(/\n\n/g, '</p><p>');
        html = '<p>' + html + '</p>';

        // Clean up empty paragraphs and fix structure
        html = html.replace(/<p><\/p>/g, '');
        html = html.replace(/<p>(<h[1-6])/g, '$1');
        html = html.replace(/(<\/h[1-6]>)<\/p>/g, '$1');
        html = html.replace(/<p>(<ul>)/g, '$1');
        html = html.replace(/(<\/ul>)<\/p>/g, '$1');
        html = html.replace(/<p>(<div)/g, '$1');
        html = html.replace(/(<\/div>)<\/p>/g, '$1');
        html = html.replace(/<p>(<table>)/g, '$1');
        html = html.replace(/(<\/table>)<\/p>/g, '$1');

        return html;
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    renderContent(html) {
        const contentArea = document.getElementById('docs-content');
        if (contentArea) {
            contentArea.innerHTML = html;
            this.addAnchorLinks();
            this.updateTableOfContents();
        }
    }

    renderError(docName) {
        const contentArea = document.getElementById('docs-content');
        if (contentArea) {
            contentArea.innerHTML = `
                <div class="error-message">
                    <h2>Documentation Not Found</h2>
                    <p>Sorry, the documentation "${docName}" could not be loaded.</p>
                    <button onclick="window.markdownRenderer.loadMarkdown('main')" class="retry-btn">Go to Main Documentation</button>
                </div>
            `;
        }
    }

    setupSidebar() {
        const sidebar = document.getElementById('docs-sidebar');
        if (!sidebar || !this.config.sidebar) return;

        // Keep the search input
        const searchInput = sidebar.querySelector('.docs-search');
        
        let sidebarHtml = '';
        if (searchInput) {
            sidebarHtml += searchInput.outerHTML;
        }

        this.config.sidebar.forEach(section => {
            sidebarHtml += `
                <div class="sidebar-section">
                    <div class="sidebar-title ${section.collapsed ? 'collapsed' : ''}" 
                         onclick="toggleSection(this)">
                        <span>${section.title}</span>
                        <i class="toggle-icon">${section.collapsed ? '▶' : '▼'}</i>
                    </div>
                    <div class="sidebar-items ${section.collapsed ? 'hidden' : ''}">
                        ${section.items.map(item => `
                            <a href="#" class="sidebar-item" 
                               onclick="navigateToDoc('${item.doc || 'main'}', '${item.anchor}')"
                               data-doc="${item.doc || 'main'}" 
                               data-anchor="${item.anchor}">
                                ${item.title}
                            </a>
                        `).join('')}
                    </div>
                </div>
            `;
        });

        sidebar.innerHTML = sidebarHtml;
        this.setupSearch();
    }

    setupSearch() {
        const searchInput = document.getElementById('docs-search');
        if (searchInput) {
            searchInput.addEventListener('input', (e) => {
                this.performSearch(e.target.value);
            });
        }
    }

    performSearch(query) {
        if (!query.trim()) {
            this.clearSearchResults();
            return;
        }

        const content = document.getElementById('docs-content');
        const text = content.textContent.toLowerCase();
        const searchQuery = query.toLowerCase();

        if (text.includes(searchQuery)) {
            this.highlightSearchResults(query);
        } else {
            this.showNoResults(query);
        }
    }

    highlightSearchResults(query) {
        const content = document.getElementById('docs-content');
        const regex = new RegExp(`(${this.escapeRegex(query)})`, 'gi');
        content.innerHTML = content.innerHTML.replace(regex, '<mark>$1</mark>');
    }

    escapeRegex(string) {
        return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    }

    clearSearchResults() {
        const content = document.getElementById('docs-content');
        content.innerHTML = content.innerHTML.replace(/<mark>(.*?)<\/mark>/g, '$1');
    }

    showNoResults(query) {
        console.log(`No results found for: ${query}`);
    }

    addAnchorLinks() {
        const headers = document.querySelectorAll('#docs-content h1, #docs-content h2, #docs-content h3');
        headers.forEach(header => {
            const id = header.textContent.toLowerCase()
                .replace(/[^a-z0-9]+/g, '-')
                .replace(/^-|-$/g, '');
            header.id = id;
            
            const anchor = document.createElement('a');
            anchor.href = `#${id}`;
            anchor.className = 'anchor-link';
            anchor.innerHTML = '#';
            header.appendChild(anchor);
        });
    }

    updateTableOfContents() {
        const toc = document.getElementById('table-of-contents');
        if (!toc) return;

        const headers = document.querySelectorAll('#docs-content h2, #docs-content h3');
        if (headers.length === 0) {
            toc.style.display = 'none';
            return;
        }

        let tocHtml = '<h3>Table of Contents</h3><ul>';
        
        headers.forEach(header => {
            const level = header.tagName.toLowerCase();
            const text = header.textContent.replace('#', '').trim();
            const id = header.id;
            
            tocHtml += `<li class="toc-${level}">
                <a href="#${id}" onclick="scrollToAnchor('#${id}')">${text}</a>
            </li>`;
        });
        
        tocHtml += '</ul>';
        toc.innerHTML = tocHtml;
        toc.style.display = 'block';
    }

    addCodeHighlighting() {
        const codeBlocks = document.querySelectorAll('code[class*="language-"]');
        codeBlocks.forEach(block => {
            this.highlightCode(block);
        });
    }

    highlightCode(block) {
        const language = block.className.match(/language-(\w+)/)?.[1];
        if (!language) return;

        let code = block.innerHTML;
        
        // 简单的语法高亮规则
        if (language === 'python') {
            code = code.replace(/\b(def|class|import|from|if|else|elif|for|while|try|except|return|yield)\b/g, '<span class="keyword">$1</span>');
            code = code.replace(/#.*$/gm, '<span class="comment">$&</span>');
            code = code.replace(/"([^"]*)"/g, '<span class="string">"$1"</span>');
            code = code.replace(/'([^']*)'/g, '<span class="string">\'$1\'</span>');
        } else if (language === 'bash') {
            code = code.replace(/^(\$|#)/gm, '<span class="prompt">$1</span>');
            code = code.replace(/\b(pip|npm|git|cd|ls|mkdir)\b/g, '<span class="command">$1</span>');
        } else if (language === 'javascript') {
            code = code.replace(/\b(function|const|let|var|if|else|for|while|return|class|import|export)\b/g, '<span class="keyword">$1</span>');
            code = code.replace(/\/\/.*$/gm, '<span class="comment">$&</span>');
            code = code.replace(/"([^"]*)"/g, '<span class="string">"$1"</span>');
            code = code.replace(/'([^']*)'/g, '<span class="string">\'$1\'</span>');
        }

        block.innerHTML = code;
    }

    addCopyButtons() {
        // Copy button functionality is handled by the global copyCode function
    }

    updateSidebar(docName) {
        const sidebarItems = document.querySelectorAll('.sidebar-item');
        sidebarItems.forEach(item => {
            item.classList.remove('active');
            if (item.dataset.doc === docName) {
                item.classList.add('active');
            }
        });
    }

    updatePageTitle(docName) {
        const section = this.config.sections?.find(s => s.id === docName);
        if (section) {
            document.title = `${section.title} - ACE3 Documentation`;
        }
    }
}

// 全局函数
function toggleSection(element) {
    const section = element.parentElement;
    const items = section.querySelector('.sidebar-items');
    const icon = element.querySelector('.toggle-icon');
    
    if (items.classList.contains('hidden')) {
        items.classList.remove('hidden');
        element.classList.remove('collapsed');
        icon.textContent = '▼';
    } else {
        items.classList.add('hidden');
        element.classList.add('collapsed');
        icon.textContent = '▶';
    }
}

function scrollToAnchor(anchor) {
    const element = document.querySelector(anchor);
    if (element) {
        element.scrollIntoView({ behavior: 'smooth' });
    }
}

function navigateToDoc(docName, anchor) {
    if (window.markdownRenderer) {
        window.markdownRenderer.loadMarkdown(docName).then(() => {
            if (anchor) {
                setTimeout(() => scrollToAnchor(anchor), 100);
            }
        });
    }
}

function copyCode(button) {
    const codeBlock = button.closest('.code-block').querySelector('code');
    const text = codeBlock.textContent;
    
    navigator.clipboard.writeText(text).then(() => {
        button.textContent = 'Copied!';
        setTimeout(() => {
            button.textContent = 'Copy';
        }, 2000);
    }).catch(err => {
        console.error('Failed to copy code:', err);
    });
}

// 初始化
document.addEventListener('DOMContentLoaded', () => {
    window.markdownRenderer = new MarkdownRenderer();
});

