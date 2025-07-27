/**
 * main.js
 * 主要JavaScript功能
 */

document.addEventListener('DOMContentLoaded', () => {
    // 处理导航栏滚动效果
    const navbar = document.querySelector('.navbar');
    
    window.addEventListener('scroll', () => {
        if (window.scrollY > 50) {
            navbar.classList.add('scrolled');
        } else {
            navbar.classList.remove('scrolled');
        }
    });
    
    // 处理轮播点击
    const dots = document.querySelectorAll('.dot');
    
    dots.forEach((dot, index) => {
        dot.addEventListener('click', () => {
            // 移除所有active类
            dots.forEach(d => d.classList.remove('active'));
            // 添加active类到当前点
            dot.classList.add('active');
            
            // 这里可以添加轮播切换逻辑
            console.log(`Switched to slide ${index + 1}`);
        });
    });
    
    // 处理语言选择器和下拉菜单
    const languageSelector = document.querySelector('.language-selector');
    const dropdown = document.querySelector('.dropdown');
    
    if (languageSelector) {
        languageSelector.addEventListener('click', () => {
            console.log('Language selector clicked');
            // 这里可以添加语言切换逻辑
        });
    }
    
    if (dropdown) {
        dropdown.addEventListener('click', () => {
            console.log('Product dropdown clicked');
            // 这里可以添加下拉菜单显示逻辑
        });
    }
});
