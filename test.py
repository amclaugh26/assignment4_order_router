from somewhat_smart_order_router import best_price_improvement

exchange, score = best_price_improvement(
    symbol="AAPL",
    side="B",
    quantity=100,
    limit_price=150.0,
    bid_price=149.8,
    ask_price=150.2,
    bid_size=300,
    ask_size=280,
)
print(f"Route to {exchange} (predicted improvement {score:.4f})")
