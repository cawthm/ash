## The Big Picture

We want to train a trading algorithm that is a modeled on next token prediction, assigning probabilities to potential future asset prices in the short term (1 to 600 seconds into the future) *PROBABILISTICALLY*, ie we want to be able to generate probabilties (rapidly) for consumption by some decision / trading engine. In a language setting, next token prediction typically involves discretization and tokenization which is then assigned to embeddings.

For prices the data is more continuous, and therefore it might not make sense to use this exact approach. On the other hand, in both the interest of speed and simplicity, it might make sense to have buckets in some continuous space (time, price, implied vol, actual / realized vol, volume, etc.).

## Our data
There is an MCP server that has been connected to this version of Claude with a database called "datawarehouse" containing two tables: options_data and stock_trades. We will want to sensibly and economically segregate, align, and consume this data as we progress.

## The use case
Once created, we will want to modularly connect the model to a streaming dataset which will arrive in some form.

## Principles 
- Speed
- Readibility
