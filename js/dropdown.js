/**
 * dropdown.js
 * 实现导航栏下拉菜单功能
 * 为ACE3英文网站添加产品下拉菜单
 */

document.addEventListener('DOMContentLoaded', () => {
    // 初始化产品下拉菜单
    initProductDropdown();
});

/**
 * 初始化产品下拉菜单
 */
function initProductDropdown() {
    const productDropdown = document.querySelector('.dropdown');
    if (!productDropdown) return;
    
    // 创建下拉菜单容器
    const dropdownContent = document.createElement('div');
    dropdownContent.className = 'dropdown-content product-dropdown-content';
    
    // 添加产品选项
    const products = [
        { id: 'ace3cloud', name: 'ACE3Cloud', description: 'Cloud AI Infrastructure' },
        { id: 'ace3suite', name: 'ACE3Suite', description: 'High-Performance Inference' },
        { id: 'ace3llm', name: 'ACE3LLM', description: 'LLM Optimization' },
        { id: 'onediff', name: 'OneDiff', description: 'Image Generation Acceleration' },
        { id: 'ace3brain', name: 'ACE3Brain', description: 'AI Development Platform' }
    ];
    
    products.forEach(product => {
        const productOption = document.createElement('a');
        productOption.href = 'products.html';
        
        // 创建产品信息容器
        const productInfo = document.createElement('div');
        productInfo.className = 'product-info';
        
        // 添加产品名称
        const productName = document.createElement('span');
        productName.className = 'product-name';
        productName.textContent = product.name;
        
        // 添加产品描述
        const productDesc = document.createElement('span');
        productDesc.className = 'product-description';
        productDesc.textContent = product.description;
        
        // 组装产品信息
        productInfo.appendChild(productName);
        productInfo.appendChild(productDesc);
        productOption.appendChild(productInfo);
        
        dropdownContent.appendChild(productOption);
    });
    
    // 将下拉菜单添加到产品下拉区域
    productDropdown.appendChild(dropdownContent);
    
    // 添加点击事件以显示/隐藏下拉菜单
    productDropdown.addEventListener('click', (e) => {
        e.stopPropagation();
        dropdownContent.classList.toggle('show');
        
        // 关闭其他可能打开的下拉菜单
        const otherDropdowns = document.querySelectorAll('.dropdown-content');
        otherDropdowns.forEach(dropdown => {
            if (dropdown !== dropdownContent && dropdown.classList.contains('show')) {
                dropdown.classList.remove('show');
            }
        });
    });
}

// 点击页面其他区域关闭所有下拉菜单
document.addEventListener('click', () => {
    const dropdowns = document.querySelectorAll('.dropdown-content');
    dropdowns.forEach(dropdown => {
        if (dropdown.classList.contains('show')) {
            dropdown.classList.remove('show');
        }
    });
});
