/**
 * trajectory-flow.js
 * 实现轨迹流动效果的JavaScript代码
 * 为ACE3英文网站创建类似硅基流动网站的背景效果
 */

// 轨迹点类
class TrajectoryPoint {
    constructor(x, y, size, speed, angle, color) {
        this.x = x;
        this.y = y;
        this.size = size;
        this.speed = speed;
        this.angle = angle; // 移动角度（弧度）
        this.color = color;
        this.opacity = Math.random() * 0.5 + 0.1; // 随机透明度
        this.element = null;
        this.createPointElement();
    }

    // 创建轨迹点DOM元素
    createPointElement() {
        this.element = document.createElement('div');
        this.element.className = 'trajectory-point';
        this.element.style.width = `${this.size}px`;
        this.element.style.height = `${this.size}px`;
        this.element.style.backgroundColor = `rgba(56, 189, 248, ${this.opacity})`;
        this.element.style.left = `${this.x}px`;
        this.element.style.top = `${this.y}px`;
        
        // 添加到背景容器
        const container = document.querySelector('.background-container');
        if (container) {
            container.appendChild(this.element);
        }
    }

    // 更新轨迹点位置
    update() {
        // 根据角度和速度计算新位置
        this.x += Math.cos(this.angle) * this.speed;
        this.y += Math.sin(this.angle) * this.speed;
        
        // 更新DOM元素位置
        if (this.element) {
            this.element.style.left = `${this.x}px`;
            this.element.style.top = `${this.y}px`;
        }
        
        // 检查是否超出屏幕边界
        const screenWidth = window.innerWidth;
        const screenHeight = window.innerHeight;
        
        // 如果超出边界，从另一侧重新进入
        if (this.x < -this.size) this.x = screenWidth + this.size;
        if (this.x > screenWidth + this.size) this.x = -this.size;
        if (this.y < -this.size) this.y = screenHeight + this.size;
        if (this.y > screenHeight + this.size) this.y = -this.size;
    }
    
    // 移除轨迹点
    remove() {
        if (this.element && this.element.parentNode) {
            this.element.parentNode.removeChild(this.element);
        }
    }
}

// 轨迹流动管理器
class TrajectoryFlowManager {
    constructor(pointCount = 30) {
        this.points = [];
        this.pointCount = pointCount;
        this.isRunning = false;
        this.animationFrameId = null;
        
        // 绑定方法到实例
        this.animate = this.animate.bind(this);
        this.handleResize = this.handleResize.bind(this);
        
        // 添加窗口大小变化监听
        window.addEventListener('resize', this.handleResize);
    }
    
    // 初始化轨迹点
    initialize() {
        // 清除现有点
        this.clear();
        
        const screenWidth = window.innerWidth;
        const screenHeight = window.innerHeight;
        
        // 创建新的轨迹点
        for (let i = 0; i < this.pointCount; i++) {
            const size = Math.random() * 10 + 5; // 5-15px
            const x = Math.random() * screenWidth;
            const y = Math.random() * screenHeight;
            const speed = Math.random() * 0.5 + 0.2; // 0.2-0.7
            const angle = Math.random() * Math.PI * 2; // 0-2π
            const color = 'rgba(56, 189, 248, 0.3)'; // 蓝色
            
            const point = new TrajectoryPoint(x, y, size, speed, angle, color);
            this.points.push(point);
        }
    }
    
    // 开始动画
    start() {
        if (!this.isRunning) {
            this.isRunning = true;
            this.animate();
        }
    }
    
    // 停止动画
    stop() {
        this.isRunning = false;
        if (this.animationFrameId) {
            cancelAnimationFrame(this.animationFrameId);
            this.animationFrameId = null;
        }
    }
    
    // 清除所有轨迹点
    clear() {
        this.points.forEach(point => point.remove());
        this.points = [];
    }
    
    // 动画循环
    animate() {
        if (!this.isRunning) return;
        
        // 更新所有轨迹点
        this.points.forEach(point => point.update());
        
        // 继续下一帧
        this.animationFrameId = requestAnimationFrame(this.animate);
    }
    
    // 处理窗口大小变化
    handleResize() {
        // 重新初始化以适应新的窗口大小
        this.initialize();
        
        // 如果动画正在运行，确保继续
        if (this.isRunning) {
            this.start();
        }
    }
}

// 当DOM加载完成后初始化
document.addEventListener('DOMContentLoaded', () => {
    // 创建轨迹流动管理器实例
    const trajectoryFlow = new TrajectoryFlowManager(40); // 40个轨迹点
    
    // 初始化并开始动画
    trajectoryFlow.initialize();
    trajectoryFlow.start();
    
    // 将实例暴露到全局，方便调试
    window.trajectoryFlow = trajectoryFlow;
});
