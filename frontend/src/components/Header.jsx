function Header() {
    return (
        <header className="header">
            <div className="header-inner">
                <a href="/" className="logo" onClick={(e) => { e.preventDefault(); window.location.reload(); }}>
                    <div className="logo-icon">🛡️</div>
                    <span className="logo-text">DeepGuard AI</span>
                    <span className="logo-badge">Beta</span>
                </a>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <span style={{ fontSize: '0.85rem', color: 'var(--gray-400)' }}>MFCC + NLP Engine</span>
                </div>
            </div>
        </header>
    )
}

export default Header
