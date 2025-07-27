/**
 * auth.js - ACE3 Authentication Module
 * 处理用户认证相关功能，支持Google OAuth和邮箱登录
 */

// 后端API基础URL - 根据环境自动切换
const API_BASE_URL = window.location.hostname === 'localhost' 
    ? 'http://localhost:5000/api/auth'
    : 'https://api.ace3.ai/api/auth';

// Google OAuth配置
const GOOGLE_CLIENT_ID = 'your-google-client-id.apps.googleusercontent.com';

// 当前用户信息
let currentUser = null;

/**
 * 初始化认证系统
 */
function initAuth() {
    console.log('Initializing ACE3 Auth System...');
    
    // 检查用户是否已登录
    checkAuthStatus();
    
    // 绑定表单事件
    bindAuthEvents();
    
    // 初始化Google Sign-In
    initGoogleSignIn();
}

/**
 * 检查用户认证状态
 */
async function checkAuthStatus() {
    try {
        const response = await fetch(`${API_BASE_URL}/me`, {
            method: 'GET',
            credentials: 'include',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        
        if (response.ok) {
            const user = await response.json();
            currentUser = user;
            console.log('User authenticated:', user.email);
            
            // 如果用户已登录且在登录页面，重定向到仪表板
            if (window.location.pathname.includes('login.html')) {
                showMessage('Already logged in, redirecting to dashboard...', 'success');
                setTimeout(() => {
                    window.location.href = 'dashboard.html';
                }, 1000);
            }
        } else {
            console.log('User not authenticated');
        }
    } catch (error) {
        console.log('Auth check failed:', error);
    }
}

/**
 * 绑定认证相关事件
 */
function bindAuthEvents() {
    // 邮箱登录表单
    const emailForm = document.getElementById('email-login-form');
    if (emailForm) {
        emailForm.addEventListener('submit', handleEmailLogin);
    }
    
    // Google登录按钮
    const googleBtn = document.getElementById('google-login-btn');
    if (googleBtn) {
        googleBtn.addEventListener('click', handleGoogleLogin);
    }
    
    // 注册链接
    const signupLink = document.getElementById('signup-link');
    if (signupLink) {
        signupLink.addEventListener('click', handleSignupClick);
    }
    
    // 忘记密码链接
    const forgotPasswordLink = document.querySelector('a[href="#"]:not(#signup-link)');
    if (forgotPasswordLink && forgotPasswordLink.textContent.includes('Forgot')) {
        forgotPasswordLink.addEventListener('click', handleForgotPassword);
    }
}

/**
 * 处理邮箱登录
 */
async function handleEmailLogin(event) {
    event.preventDefault();
    
    const email = document.getElementById('email').value.trim();
    const password = document.getElementById('password').value;
    
    if (!email || !password) {
        showMessage('Please enter both email and password', 'error');
        return;
    }
    
    showMessage('Signing in...', 'info');
    
    try {
        // 演示模式：使用Google登录接口模拟邮箱登录
        const response = await fetch(`${API_BASE_URL}/google-login`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            credentials: 'include',
            body: JSON.stringify({
                token: 'demo_email_token_' + Date.now(),
                email: email,
                name: email.split('@')[0].charAt(0).toUpperCase() + email.split('@')[0].slice(1),
                picture: `https://ui-avatars.com/api/?name=${encodeURIComponent(email.split('@')[0])}&background=FF6B35&color=fff`
            })
        });
        
        const result = await response.json();
        
        if (response.ok && result.success) {
            currentUser = result.user;
            showMessage('Login successful! Redirecting to dashboard...', 'success');
            
            // 保存登录状态到localStorage（可选）
            localStorage.setItem('ace3_user', JSON.stringify(result.user));
            
            setTimeout(() => {
                window.location.href = 'dashboard.html';
            }, 1500);
        } else {
            showMessage(result.error || 'Login failed. Please try again.', 'error');
        }
    } catch (error) {
        console.error('Email login error:', error);
        showMessage('Network error. Please check your connection and try again.', 'error');
    }
}

/**
 * 初始化Google Sign-In
 */
function initGoogleSignIn() {
    // 检查Google Sign-In API是否加载
    if (typeof google !== 'undefined' && google.accounts) {
        console.log('Google Sign-In API loaded successfully');
        
        // 初始化Google Identity Services
        google.accounts.id.initialize({
            client_id: GOOGLE_CLIENT_ID,
            callback: handleCredentialResponse,
            auto_select: false,
            cancel_on_tap_outside: true
        });
        
        // 可选：渲染Google One Tap
        // google.accounts.id.prompt();
    } else {
        console.log('Google Sign-In API not loaded, using fallback');
    }
}

/**
 * 处理Google OAuth回调
 */
window.handleCredentialResponse = async function(response) {
    console.log('Google credential response received');
    
    try {
        showMessage('Verifying Google credentials...', 'info');
        
        const result = await fetch(`${API_BASE_URL}/google-login`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            credentials: 'include',
            body: JSON.stringify({
                token: response.credential
            })
        });
        
        const data = await result.json();
        
        if (result.ok && data.success) {
            currentUser = data.user;
            showMessage('Google login successful! Redirecting...', 'success');
            
            // 保存登录状态
            localStorage.setItem('ace3_user', JSON.stringify(data.user));
            
            setTimeout(() => {
                window.location.href = 'dashboard.html';
            }, 1500);
        } else {
            showMessage(data.error || 'Google login failed', 'error');
        }
    } catch (error) {
        console.error('Google login error:', error);
        showMessage('Google login error. Please try again.', 'error');
    }
};

/**
 * 处理Google登录按钮点击（备用方案）
 */
function handleGoogleLogin() {
    console.log('Google login button clicked');
    
    // 尝试使用Google Identity Services
    if (typeof google !== 'undefined' && google.accounts) {
        google.accounts.id.prompt((notification) => {
            if (notification.isNotDisplayed() || notification.isSkippedMoment()) {
                // 如果One Tap不可用，使用演示模式
                handleDemoGoogleLogin();
            }
        });
    } else {
        // 备用演示模式
        handleDemoGoogleLogin();
    }
}

/**
 * 演示模式的Google登录
 */
async function handleDemoGoogleLogin() {
    const demoUser = {
        email: 'demo.user@gmail.com',
        name: 'Demo User',
        picture: 'https://ui-avatars.com/api/?name=Demo+User&background=4285F4&color=fff'
    };
    
    showMessage('Demo: Signing in with Google...', 'info');
    
    try {
        const response = await fetch(`${API_BASE_URL}/google-login`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            credentials: 'include',
            body: JSON.stringify({
                token: 'demo_google_token_' + Date.now(),
                email: demoUser.email,
                name: demoUser.name,
                picture: demoUser.picture
            })
        });
        
        const result = await response.json();
        
        if (response.ok && result.success) {
            currentUser = result.user;
            showMessage('Google login successful! Redirecting...', 'success');
            
            localStorage.setItem('ace3_user', JSON.stringify(result.user));
            
            setTimeout(() => {
                window.location.href = 'dashboard.html';
            }, 1500);
        } else {
            showMessage(result.error || 'Google login failed', 'error');
        }
    } catch (error) {
        console.error('Demo Google login error:', error);
        showMessage('Network error. Please try again.', 'error');
    }
}

/**
 * 处理注册链接点击
 */
function handleSignupClick(event) {
    event.preventDefault();
    showMessage('Registration is coming soon! Please use Google login for now.', 'info');
}

/**
 * 处理忘记密码
 */
function handleForgotPassword(event) {
    event.preventDefault();
    showMessage('Password reset feature is coming soon! Please use Google login.', 'info');
}

/**
 * 用户登出
 */
async function logout() {
    try {
        showMessage('Signing out...', 'info');
        
        const response = await fetch(`${API_BASE_URL}/logout`, {
            method: 'POST',
            credentials: 'include',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        
        if (response.ok) {
            currentUser = null;
            localStorage.removeItem('ace3_user');
            
            // 如果有Google Sign-In，也要登出
            if (typeof google !== 'undefined' && google.accounts) {
                google.accounts.id.disableAutoSelect();
            }
            
            showMessage('Logged out successfully', 'success');
            setTimeout(() => {
                window.location.href = 'login.html';
            }, 1000);
        } else {
            throw new Error('Logout failed');
        }
    } catch (error) {
        console.error('Logout error:', error);
        // 即使出错也清除本地状态
        currentUser = null;
        localStorage.removeItem('ace3_user');
        window.location.href = 'login.html';
    }
}

/**
 * 显示消息
 */
function showMessage(message, type = 'info') {
    console.log(`[${type.toUpperCase()}] ${message}`);
    
    const errorDiv = document.getElementById('login-error');
    const successDiv = document.getElementById('login-success');
    
    // 隐藏所有消息
    if (errorDiv) errorDiv.style.display = 'none';
    if (successDiv) successDiv.style.display = 'none';
    
    // 显示对应类型的消息
    if (type === 'error' && errorDiv) {
        errorDiv.textContent = message;
        errorDiv.style.display = 'block';
        
        // 自动隐藏错误消息
        setTimeout(() => {
            errorDiv.style.display = 'none';
        }, 5000);
    } else if (type === 'success' && successDiv) {
        successDiv.textContent = message;
        successDiv.style.display = 'block';
    } else if (type === 'info') {
        // 对于info消息，可以使用临时通知
        const notification = document.createElement('div');
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: #4299E1;
            color: white;
            padding: 1rem 1.5rem;
            border-radius: 0.5rem;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            z-index: 1000;
            font-size: 0.875rem;
            max-width: 300px;
        `;
        notification.textContent = message;
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.remove();
        }, 3000);
    }
}

/**
 * 获取当前用户信息
 */
function getCurrentUser() {
    return currentUser || JSON.parse(localStorage.getItem('ace3_user') || 'null');
}

/**
 * 检查用户是否已登录
 */
function isAuthenticated() {
    return getCurrentUser() !== null;
}

/**
 * 获取用户API密钥
 */
async function getApiKey() {
    const user = getCurrentUser();
    if (!user) return null;
    
    try {
        const response = await fetch(`${API_BASE_URL}/me`, {
            method: 'GET',
            credentials: 'include'
        });
        
        if (response.ok) {
            const userData = await response.json();
            return userData.api_key;
        }
    } catch (error) {
        console.error('Failed to get API key:', error);
    }
    
    return null;
}

/**
 * 重新生成API密钥
 */
async function regenerateApiKey() {
    try {
        const response = await fetch(`${API_BASE_URL}/regenerate-api-key`, {
            method: 'POST',
            credentials: 'include'
        });
        
        if (response.ok) {
            const result = await response.json();
            return result.api_key;
        } else {
            throw new Error('Failed to regenerate API key');
        }
    } catch (error) {
        console.error('API key regeneration error:', error);
        throw error;
    }
}

// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', initAuth);

// 导出函数供其他脚本使用
window.authModule = {
    getCurrentUser,
    isAuthenticated,
    logout,
    showMessage,
    getApiKey,
    regenerateApiKey
};

console.log('ACE3 Auth Module loaded successfully');

