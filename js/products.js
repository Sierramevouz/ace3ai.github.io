/**
 * products.js
 * 实现产品页面的筛选和详情展开/折叠功能
 */

document.addEventListener('DOMContentLoaded', function() {
    // 产品筛选功能
    const filterButtons = document.querySelectorAll('.filter-button');
    const productCards = document.querySelectorAll('.product-card');
    
    // 为所有筛选按钮添加点击事件
    filterButtons.forEach(button => {
        button.addEventListener('click', function() {
            // 移除所有按钮的active类
            filterButtons.forEach(btn => btn.classList.remove('active'));
            
            // 为当前点击的按钮添加active类
            this.classList.add('active');
            
            // 获取筛选类别
            const filterValue = this.getAttribute('data-filter');
            
            // 筛选产品卡片
            productCards.forEach(card => {
                if (filterValue === 'all' || card.getAttribute('data-category') === filterValue) {
                    card.style.display = 'block';
                } else {
                    card.style.display = 'none';
                }
            });
        });
    });
    
    // 详情展开/折叠功能
    const toggleButtons = document.querySelectorAll('.toggle-details');
    
    toggleButtons.forEach(button => {
        button.addEventListener('click', function() {
            // 获取当前产品卡片
            const productCard = this.closest('.product-card');
            
            // 切换expanded类
            productCard.classList.toggle('expanded');
            
            // 更新按钮文本
            const toggleIcon = this.querySelector('.toggle-icon');
            if (productCard.classList.contains('expanded')) {
                this.innerHTML = 'Hide Details <span class="toggle-icon">▲</span>';
            } else {
                this.innerHTML = 'Details <span class="toggle-icon">▼</span>';
            }
        });
    });
});
