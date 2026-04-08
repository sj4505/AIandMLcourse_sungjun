const crypto = require('crypto');

// Supabase 클라이언트 모킹
const mockUpdate = jest.fn().mockReturnValue({ eq: jest.fn().mockResolvedValue({ error: null }) });
jest.mock('@supabase/supabase-js', () => ({
  createClient: () => ({ from: () => ({ update: mockUpdate }) })
}));

const handler = require('../api/webhook/polar');

const WEBHOOK_SECRET = 'whsec_dGVzdHNlY3JldA=='; // whsec_ + base64('testsecret')
const SECRET_RAW = 'dGVzdHNlY3JldA==';

function makeRequest(body, overrideSignature) {
  const msgId = 'msg_test_001';
  const msgTimestamp = String(Math.floor(Date.now() / 1000));
  const toSign = `${msgId}.${msgTimestamp}.${body}`;
  const secretBytes = Buffer.from(SECRET_RAW, 'base64');
  const sig = crypto.createHmac('sha256', secretBytes).update(toSign).digest('base64');

  const req = {
    method: 'POST',
    headers: {
      'webhook-id': msgId,
      'webhook-timestamp': msgTimestamp,
      'webhook-signature': overrideSignature ?? `v1,${sig}`,
    },
    on: jest.fn((event, cb) => {
      if (event === 'data') cb(body);
      if (event === 'end') cb();
      return req;
    }),
  };
  const res = {
    status: jest.fn().mockReturnThis(),
    json: jest.fn().mockReturnThis(),
    end: jest.fn(),
  };
  return { req, res };
}

beforeEach(() => {
  process.env.POLAR_WEBHOOK_SECRET = WEBHOOK_SECRET;
  process.env.SUPABASE_URL = 'https://test.supabase.co';
  process.env.SUPABASE_SERVICE_ROLE_KEY = 'service-role-key';
  mockUpdate.mockClear();
});

test('405 for non-POST method', async () => {
  const req = { method: 'GET' };
  const res = { status: jest.fn().mockReturnThis(), end: jest.fn() };
  await handler(req, res);
  expect(res.status).toHaveBeenCalledWith(405);
});

test('401 for invalid signature', async () => {
  const { req, res } = makeRequest('{}', 'v1,invalidsignature==');
  await handler(req, res);
  expect(res.status).toHaveBeenCalledWith(401);
});

test('200 and sets active for subscription.active', async () => {
  const body = JSON.stringify({
    type: 'subscription.active',
    data: { id: 'sub_abc', customer: { email: 'user@example.com' } },
  });
  const { req, res } = makeRequest(body);
  await handler(req, res);
  expect(res.status).toHaveBeenCalledWith(200);
  expect(mockUpdate).toHaveBeenCalledWith(
    expect.objectContaining({ subscription_status: 'active' })
  );
});

test('200 and sets inactive for subscription.canceled', async () => {
  const body = JSON.stringify({
    type: 'subscription.canceled',
    data: { id: 'sub_abc', customer: { email: 'user@example.com' } },
  });
  const { req, res } = makeRequest(body);
  await handler(req, res);
  expect(res.status).toHaveBeenCalledWith(200);
  expect(mockUpdate).toHaveBeenCalledWith(
    expect.objectContaining({ subscription_status: 'inactive' })
  );
});

test('200 with no-op when customer email missing', async () => {
  const body = JSON.stringify({
    type: 'subscription.active',
    data: { id: 'sub_abc' },
  });
  const { req, res } = makeRequest(body);
  await handler(req, res);
  expect(res.status).toHaveBeenCalledWith(200);
  expect(mockUpdate).not.toHaveBeenCalled();
});
