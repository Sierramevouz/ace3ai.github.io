/**
 * dashboard.js - ACE3 Dashboard Functionality
 * 处理dashboard页面的交互和数据展示
 */

// Dashboard状态
let dashboardData = {
    user: null,
    stats: {
        apiCalls: 1247,
        tokensUsed: 2400000,
        avgResponseTime: 127,
        successRate: 99.8
    },
    usage: [],
    apiKey: null
};

/**
 * 初始化Dashboard
 */
function initDashboard() {
    console.log('Initializing ACE3 Dashboard...');
    
    // 检查用户认证状态
    if (!window.authModule.isAuthenticated()) {
        console.log('User not authenticated, redirecting to login');
        window.location.href = 'login.html';
        return;
    }
    
    // 加载用户信息
    loadUserInfo();
    
    // 绑定事件
    bindDashboardEvents();
    
    // 初始化图表
    initCharts();
    
    // 加载数据
    loadDashboardData();
    
    // 设置定时刷新
    setInterval(refreshStats, 30000); // 每30秒刷新一次
}

/**
 * 加载用户信息
 */
function loadUserInfo() {
    const user = window.authModule.getCurrentUser();
    if (user) {
        dashboardData.user = user;
        
        // 更新UI
        const userAvatar = document.getElementById('user-avatar');
        const userName = document.getElementById('user-name');
        
        if (userAvatar && user.picture) {
            userAvatar.src = user.picture;
            userAvatar.alt = user.name;
        } else if (userAvatar) {
            userAvatar.src = `https://ui-avatars.com/api/?name=${encodeURIComponent(user.name)}&background=FF6B35&color=fff`;
        }
        
        if (userName) {
            userName.textContent = user.name;
        }
        
        // 更新设置页面的表单
        const settingsName = document.getElementById('settings-name');
        const settingsEmail = document.getElementById('settings-email');
        
        if (settingsName) settingsName.value = user.name;
        if (settingsEmail) settingsEmail.value = user.email;
    }
}

/**
 * 绑定Dashboard事件
 */
function bindDashboardEvents() {
    // 侧边栏导航
    const sidebarLinks = document.querySelectorAll('.sidebar-link');
    sidebarLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const section = this.getAttribute('data-section');
            showSection(section);
            
            // 更新活动状态
            sidebarLinks.forEach(l => l.classList.remove('active'));
            this.classList.add('active');
        });
    });
    
    // 登出按钮
    const logoutBtn = document.getElementById('logout-btn');
    if (logoutBtn) {
        logoutBtn.addEventListener('click', function() {
            if (confirm('Are you sure you want to logout?')) {
                window.authModule.logout();
            }
        });
    }
    
    // API密钥相关按钮
    const regenerateKeyBtn = document.getElementById('regenerate-key-btn');
    const copyKeyBtn = document.getElementById('copy-key-btn');
    
    if (regenerateKeyBtn) {
        regenerateKeyBtn.addEventListener('click', regenerateApiKey);
    }
    
    if (copyKeyBtn) {
        copyKeyBtn.addEventListener('click', copyApiKey);
    }
}

/**
 * 显示指定的section
 */
function showSection(sectionName) {
    // 隐藏所有section
    const sections = document.querySelectorAll('.dashboard-section');
    sections.forEach(section => {
        section.style.display = 'none';
    });
    
    // 显示指定section
    const targetSection = document.getElementById(`${sectionName}-section`);
    if (targetSection) {
        targetSection.style.display = 'block';
        
        // 根据section加载特定数据
        switch(sectionName) {
            case 'api-keys':
                loadApiKey();
                break;
            case 'usage':
                loadUsageData();
                break;
            case 'models':
                loadModels();
                break;
            case 'billing':
                loadBillingInfo();
                break;
        }
    }
}

/**
 * 初始化图表
 */
function initCharts() {
    // API使用趋势图表
    const usageCtx = document.getElementById('usage-chart');
    if (usageCtx) {
        new Chart(usageCtx, {
            type: 'line',
            data: {
                labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
                datasets: [{
                    label: 'API Calls',
                    data: [850, 920, 1100, 1050, 1180, 1247],
                    borderColor: '#FF6B35',
                    backgroundColor: 'rgba(255, 107, 53, 0.1)',
                    tension: 0.4,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        grid: {
                            color: '#E2E8F0'
                        }
                    },
                    x: {
                        grid: {
                            display: false
                        }
                    }
                }
            }
        });
    }
    
    // 模型使用分布图表
    const modelCtx = document.getElementById('model-chart');
    if (modelCtx) {
        new Chart(modelCtx, {
            type: 'doughnut',
            data: {
                labels: ['Inference v1.0', 'Training v1.0', 'Custom Models'],
                datasets: [{
                    data: [65, 25, 10],
                    backgroundColor: ['#FF6B35', '#4299E1', '#48BB78'],
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            padding: 20,
                            usePointStyle: true
                        }
                    }
                }
            }
        });
    }
}

/**
 * 加载Dashboard数据
 */
async function loadDashboardData() {
    try {
        // 这里可以调用真实的API获取数据
        // const response = await fetch('/api/dashboard/stats');
        // const data = await response.json();
        
        // 演示数据
        updateStats(dashboardData.stats);
        
    } catch (error) {
        console.error('Failed to load dashboard data:', error);
        window.authModule.showMessage('Failed to load dashboard data', 'error');
    }
}

/**
 * 更新统计数据
 */
function updateStats(stats) {
    const elements = {
        'api-calls-count': stats.apiCalls.toLocaleString(),
        'tokens-used': (stats.tokensUsed / 1000000).toFixed(1) + 'M',
        'avg-response-time': stats.avgResponseTime + 'ms',
        'success-rate': stats.successRate + '%'
    };
    
    Object.entries(elements).forEach(([id, value]) => {
        const element = document.getElementById(id);
        if (element) {
            element.textContent = value;
        }
    });
}

/**
 * 刷新统计数据
 */
async function refreshStats() {
    try {
        // 模拟数据更新
        dashboardData.stats.apiCalls += Math.floor(Math.random() * 10);
        dashboardData.stats.tokensUsed += Math.floor(Math.random() * 1000);
        
        updateStats(dashboardData.stats);
    } catch (error) {
        console.error('Failed to refresh stats:', error);
    }
}

/**
 * 加载API密钥
 */
async function loadApiKey() {
    try {
        const apiKey = await window.authModule.getApiKey();
        if (apiKey) {
            dashboardData.apiKey = apiKey;
            const apiKeyElement = document.getElementById('api-key-value');
            if (apiKeyElement) {
                // 显示部分密钥，其余用*号隐藏
                const maskedKey = apiKey.substring(0, 8) + '...' + apiKey.substring(apiKey.length - 8);
                apiKeyElement.textContent = maskedKey;
            }
        } else {
            // 生成演示API密钥
            const demoKey = 'ace3_' + Math.random().toString(36).substring(2, 15) + Math.random().toString(36).substring(2, 15);
            dashboardData.apiKey = demoKey;
            const apiKeyElement = document.getElementById('api-key-value');
            if (apiKeyElement) {
                const maskedKey = demoKey.substring(0, 8) + '...' + demoKey.substring(demoKey.length - 8);
                apiKeyElement.textContent = maskedKey;
            }
        }
    } catch (error) {
        console.error('Failed to load API key:', error);
        window.authModule.showMessage('Failed to load API key', 'error');
    }
}

/**
 * 重新生成API密钥
 */
async function regenerateApiKey() {
    if (!confirm('Are you sure you want to regenerate your API key? This will invalidate the current key.')) {
        return;
    }
    
    try {
        const newApiKey = await window.authModule.regenerateApiKey();
        if (newApiKey) {
            dashboardData.apiKey = newApiKey;
            const apiKeyElement = document.getElementById('api-key-value');
            if (apiKeyElement) {
                const maskedKey = newApiKey.substring(0, 8) + '...' + newApiKey.substring(newApiKey.length - 8);
                apiKeyElement.textContent = maskedKey;
            }
            window.authModule.showMessage('API key regenerated successfully', 'success');
        }
    } catch (error) {
        console.error('Failed to regenerate API key:', error);
        window.authModule.showMessage('Failed to regenerate API key', 'error');
    }
}

/**
 * 复制API密钥
 */
async function copyApiKey() {
    if (!dashboardData.apiKey) {
        window.authModule.showMessage('No API key available', 'error');
        return;
    }
    
    try {
        await navigator.clipboard.writeText(dashboardData.apiKey);
        window.authModule.showMessage('API key copied to clipboard', 'success');
        
        // 更新按钮文本
        const copyBtn = document.getElementById('copy-key-btn');
        if (copyBtn) {
            const originalHTML = copyBtn.innerHTML;
            copyBtn.innerHTML = '<i class="fas fa-check"></i>';
            setTimeout(() => {
                copyBtn.innerHTML = originalHTML;
            }, 2000);
        }
    } catch (error) {
        console.error('Failed to copy API key:', error);
        window.authModule.showMessage('Failed to copy API key', 'error');
    }
}

/**
 * 加载使用数据
 */
async function loadUsageData() {
    try {
        // 生成演示使用数据
        const usageData = [];
        for (let i = 0; i < 10; i++) {
            usageData.push({
                timestamp: new Date(Date.now() - i * 3600000).toLocaleString(),
                endpoint: Math.random() > 0.5 ? '/api/inference' : '/api/training',
                tokens: Math.floor(Math.random() * 1000) + 100,
                status: Math.random() > 0.1 ? 'Success' : 'Error'
            });
        }
        
        const tableBody = document.getElementById('usage-table-body');
        if (tableBody) {
            tableBody.innerHTML = usageData.map(row => `
                <tr style="border-bottom: 1px solid #E2E8F0;">
                    <td style="padding: 0.75rem; color: #1A202C; font-size: 0.875rem;">${row.timestamp}</td>
                    <td style="padding: 0.75rem; color: #1A202C; font-size: 0.875rem;"><code>${row.endpoint}</code></td>
                    <td style="padding: 0.75rem; color: #1A202C; font-size: 0.875rem;">${row.tokens.toLocaleString()}</td>
                    <td style="padding: 0.75rem;">
                        <span style="background: ${row.status === 'Success' ? '#C6F6D5' : '#FED7D7'}; color: ${row.status === 'Success' ? '#2F855A' : '#C53030'}; padding: 0.25rem 0.75rem; border-radius: 1rem; font-size: 0.75rem; font-weight: 600;">
                            ${row.status}
                        </span>
                    </td>
                </tr>
            `).join('');
        }
    } catch (error) {
        console.error('Failed to load usage data:', error);
        window.authModule.showMessage('Failed to load usage data', 'error');
    }
}

/**
 * 加载模型信息
 */
function loadModels() {
    // 模型信息已经在HTML中静态定义
    console.log('Models section loaded');
}

/**
 * 加载账单信息
 */
function loadBillingInfo() {
    // 账单信息已经在HTML中静态定义
    console.log('Billing section loaded');
}

// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', function() {
    // 添加侧边栏样式
    const style = document.createElement('style');
    style.textContent = `
        .sidebar-link {
            transition: all 0.3s ease;
        }
        
        .sidebar-link:hover {
            background-color: #EDF2F7;
            color: #FF6B35;
        }
        
        .sidebar-link.active {
            background-color: #FF6B35;
            color: white;
        }
        
        .sidebar-link.active:hover {
            background-color: #E55A00;
        }
        
        .stat-card {
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        
        .stat-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        }
        
        .model-card {
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        
        .model-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        }
        
        @media (max-width: 768px) {
            .dashboard-container {
                flex-direction: column;
            }
            
            .dashboard-sidebar {
                width: 100%;
                padding: 1rem 0;
            }
            
            .sidebar-nav {
                display: flex;
                overflow-x: auto;
                padding: 0 1rem;
            }
            
            .sidebar-nav ul {
                display: flex;
                gap: 0.5rem;
                min-width: max-content;
            }
            
            .sidebar-nav li {
                margin-bottom: 0;
            }
            
            .sidebar-link {
                white-space: nowrap;
                padding: 0.5rem 1rem;
            }
            
            .dashboard-main {
                padding: 1rem;
            }
            
            .stats-grid {
                grid-template-columns: 1fr;
            }
        }
    `;
    document.head.appendChild(style);
    
    // 初始化Dashboard
    initDashboard();
});

console.log('ACE3 Dashboard Module loaded successfully');

