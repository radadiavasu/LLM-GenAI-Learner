asyncio - For handling tasks concurrently means it doesn't wait for one to finish it switches tasks during wait times. In trading systems we need to handle live price feeds, check your current positions, place/cancel orders via REST API, write logs etc, now without asyncio, each of these would block next, with asyncio one event loop do all.



* we use "wait" it says, "pause this task here and let other tasks run while I wait."



websockets: A persistent two-way connection unlike REST (open connection(every time for each request) -> get answer -> close connection), WebSockets give us a persistent connection to the exchange so price and orderbook updates are pushed to us in real time. REST is too slow for this — opening a new connection for every price tick would add latency and hit rate limits. In a market-making bot, the WebSocket listener feeds live data into the decision engine, which then places orders via REST..

