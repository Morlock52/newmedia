/**
 * Security Manager Test Suite
 * Comprehensive tests for enterprise security features
 */

const SecurityManager = require('../security-manager');
const { expect } = require('chai');
const sinon = require('sinon');
const crypto = require('crypto');

describe('Security Manager - Enterprise Security Tests', () => {
  let securityManager;
  let sandbox;

  beforeEach(() => {
    sandbox = sinon.createSandbox();
    securityManager = new SecurityManager({
      oidcProvider: 'keycloak',
      oidcRealm: 'test-realm',
      oidcClientId: 'test-client',
      oidcIssuer: 'https://test-auth.local',
      vaultAddress: 'https://test-vault.local:8200',
      siemEndpoint: 'https://test-siem.local:8088',
      falcoWebhook: 'https://test-webhook.local/falco'
    });
  });

  afterEach(() => {
    sandbox.restore();
    securityManager.shutdown();
  });

  describe('Zero-Trust Architecture', () => {
    it('should validate mTLS certificates correctly', async () => {
      const validCert = generateTestCertificate();
      const result = await securityManager.certificateValidator.validateCertificate(validCert);
      
      expect(result.valid).to.be.true;
      expect(result.subject).to.exist;
      expect(result.issuer).to.exist;
    });

    it('should reject expired certificates', async () => {
      const expiredCert = generateTestCertificate({ expired: true });
      const result = await securityManager.certificateValidator.validateCertificate(expiredCert);
      
      expect(result.valid).to.be.false;
      expect(result.error).to.include('expired');
    });

    it('should enforce device attestation when enabled', async () => {
      securityManager.config.zeroTrust.deviceTrust.enabled = true;
      const cert = generateTestCertificate();
      
      const attestationStub = sandbox.stub(securityManager, 'verifyDeviceAttestation')
        .resolves({ valid: false });
      
      const result = await securityManager.certificateValidator.validateCertificate(cert);
      
      expect(result.valid).to.be.false;
      expect(result.error).to.include('attestation failed');
      expect(attestationStub.calledOnce).to.be.true;
    });

    it('should check certificate revocation', async () => {
      const cert = generateTestCertificate();
      const revocationStub = sandbox.stub(securityManager, 'checkCertificateRevocation')
        .resolves(true);
      
      const result = await securityManager.certificateValidator.validateCertificate(cert);
      
      expect(result.valid).to.be.false;
      expect(result.error).to.include('revoked');
      expect(revocationStub.calledOnce).to.be.true;
    });
  });

  describe('OIDC/OAuth2 Integration', () => {
    it('should validate JWT tokens correctly', async () => {
      const token = generateTestJWT({
        sub: 'user123',
        iss: securityManager.config.oidc.issuer,
        aud: securityManager.config.oidc.clientId,
        exp: Math.floor(Date.now() / 1000) + 3600
      });

      const result = await securityManager.tokenValidator.validateToken(token);
      
      expect(result.valid).to.be.true;
      expect(result.subject).to.equal('user123');
      expect(result.claims).to.exist;
    });

    it('should reject tokens with invalid signatures', async () => {
      const token = generateTestJWT({ invalidSignature: true });
      const result = await securityManager.tokenValidator.validateToken(token);
      
      expect(result.valid).to.be.false;
      expect(result.error).to.include('Invalid');
    });

    it('should enforce token binding in zero-trust mode', async () => {
      securityManager.config.zeroTrust.enabled = true;
      const token = generateTestJWT();
      
      const bindingStub = sandbox.stub(securityManager, 'verifyTokenBinding')
        .resolves(false);
      
      const result = await securityManager.tokenValidator.validateToken(token);
      
      expect(result.valid).to.be.false;
      expect(result.error).to.include('Token binding');
      expect(bindingStub.calledOnce).to.be.true;
    });

    it('should extract roles and scopes correctly', async () => {
      const token = generateTestJWT({
        realm_access: { roles: ['admin', 'user'] },
        scope: 'read write'
      });

      const result = await securityManager.tokenValidator.validateToken(token);
      
      expect(result.roles).to.deep.equal(['admin', 'user']);
      expect(result.scope).to.equal('read write');
    });
  });

  describe('Container Runtime Security', () => {
    it('should detect Falco rule violations', async () => {
      const event = {
        container: 'consensus-node-1',
        process: 'nc',
        operation: 'spawn',
        user: 'root'
      };

      const violationStub = sandbox.stub(securityManager, 'checkFalcoRules')
        .resolves([{ rule: 'Network Scanning', severity: 'HIGH' }]);

      await securityManager.falcoClient.processEvent(event);

      expect(violationStub.calledOnce).to.be.true;
      expect(violationStub.firstCall.args[0]).to.deep.equal(event);
    });

    it('should execute automatic remediation for critical violations', async () => {
      const event = {
        container: 'consensus-node-1',
        violations: [{ rule: 'Container Escape', severity: 'CRITICAL' }]
      };

      const remediationStub = sandbox.stub(securityManager, 'executeRemediation');
      sandbox.stub(securityManager, 'checkFalcoRules')
        .resolves(event.violations);
      sandbox.stub(securityManager, 'calculateSeverity').returns('CRITICAL');

      await securityManager.falcoClient.processEvent(event);

      expect(remediationStub.calledOnce).to.be.true;
    });

    it('should monitor syscalls in real-time', async () => {
      const container = 'consensus-node-1';
      const monitor = await securityManager.falcoClient.monitorSyscalls(container);

      expect(monitor.container).to.equal(container);
      expect(monitor.startTime).to.be.closeTo(Date.now(), 100);
      expect(monitor.syscalls).to.be.instanceof(Map);
      expect(monitor.anomalies).to.be.an('array');
    });
  });

  describe('SBOM and Vulnerability Management', () => {
    it('should generate comprehensive SBOM', async () => {
      const image = 'consensus/node:latest';
      
      sandbox.stub(securityManager, 'executeSyft').resolves({
        packages: [{ name: 'openssl', version: '1.1.1' }]
      });
      sandbox.stub(securityManager, 'executeGrype').resolves({
        vulnerabilities: []
      });
      sandbox.stub(securityManager, 'executeTrivy').resolves({
        vulnerabilities: []
      });

      const sbom = await securityManager.generateSBOM(image);

      expect(sbom.name).to.equal(image);
      expect(sbom.packages).to.have.length.greaterThan(0);
      expect(sbom.spdxVersion).to.equal('SPDX-2.3');
    });

    it('should detect critical vulnerabilities', async () => {
      const image = 'consensus/node:vulnerable';
      const criticalVuln = {
        id: 'CVE-2023-12345',
        cvss: 9.8,
        severity: 'CRITICAL'
      };

      sandbox.stub(securityManager, 'executeSyft').resolves({ packages: [] });
      sandbox.stub(securityManager, 'executeGrype').resolves({
        vulnerabilities: [criticalVuln]
      });
      sandbox.stub(securityManager, 'parseGrypeResults').returns([criticalVuln]);

      const emitSpy = sandbox.spy(securityManager, 'emit');
      const sbom = await securityManager.generateSBOM(image);

      expect(sbom.vulnerabilities).to.include(criticalVuln);
      expect(emitSpy.calledWith('critical-vulnerabilities')).to.be.true;
    });

    it('should compare results from multiple scanners', async () => {
      const image = 'consensus/node:latest';
      
      sandbox.stub(securityManager, 'executeSyft').resolves({ packages: [] });
      sandbox.stub(securityManager, 'executeGrype').resolves({
        vulnerabilities: [{ id: 'CVE-2023-1', cvss: 7.5 }]
      });
      sandbox.stub(securityManager, 'executeTrivy').resolves({
        vulnerabilities: [{ id: 'CVE-2023-2', cvss: 8.0 }]
      });

      const sbom = await securityManager.generateSBOM(image);

      expect(sbom.vulnerabilities).to.exist;
      expect(sbom.trivyVulnerabilities).to.exist;
    });
  });

  describe('Network Policies and Micro-segmentation', () => {
    it('should generate correct network policies', async () => {
      const service = {
        name: 'consensus-api',
        namespace: 'consensus-system',
        labels: { app: 'consensus-api' },
        allowedIngress: [{
          namespaceLabels: { name: 'frontend' },
          podLabels: { app: 'api-gateway' },
          ports: [{ port: 8545, protocol: 'TCP' }]
        }],
        allowedEgress: [{
          namespaceLabels: { name: 'consensus-system' },
          podLabels: { app: 'consensus-node' },
          ports: [{ port: 30303, protocol: 'TCP' }]
        }]
      };

      const policies = await securityManager.applyNetworkPolicies(service);

      expect(policies.metadata.name).to.equal('consensus-api-network-policy');
      expect(policies.spec.ingress).to.have.length(1);
      expect(policies.spec.egress).to.have.length(1);
      expect(policies.spec.policyTypes).to.include('Ingress', 'Egress');
    });

    it('should enforce default deny when enabled', async () => {
      securityManager.config.networkPolicies.defaultDeny = true;
      
      const service = {
        name: 'test-service',
        namespace: 'test-ns',
        labels: { app: 'test' }
      };

      const policies = await securityManager.applyNetworkPolicies(service);

      expect(policies.spec.ingress).to.be.empty;
      expect(policies.spec.egress).to.be.empty;
    });

    it('should generate Cilium L7 policies', async () => {
      const service = {
        name: 'consensus-api',
        namespace: 'consensus-system',
        labels: { app: 'consensus-api' }
      };

      const ciliumStub = sandbox.stub(securityManager, 'generateCiliumPolicy')
        .returns({ kind: 'CiliumNetworkPolicy' });
      const applyStub = sandbox.stub(securityManager, 'applyCiliumPolicy')
        .resolves();

      await securityManager.applyNetworkPolicies(service);

      expect(ciliumStub.calledOnce).to.be.true;
      expect(applyStub.calledOnce).to.be.true;
    });
  });

  describe('HashiCorp Vault Integration', () => {
    it('should store secrets securely', async () => {
      const mockVault = {
        write: sandbox.stub().resolves({ data: { version: 1 } })
      };
      securityManager.vault = mockVault;

      const secret = {
        apiKey: 'secret-key-123',
        dbPassword: 'super-secret-pass'
      };

      const encryptStub = sandbox.stub(securityManager, 'encryptSensitiveData')
        .resolves(secret);
      const auditStub = sandbox.stub(securityManager, 'auditSecretAccess')
        .resolves();

      const result = await securityManager.storeSecret('consensus/api', secret);

      expect(encryptStub.calledWith(secret)).to.be.true;
      expect(mockVault.write.calledOnce).to.be.true;
      expect(auditStub.calledWith('write', 'consensus/api', 'success')).to.be.true;
    });

    it('should retrieve and decrypt secrets', async () => {
      const encryptedData = { data: 'encrypted' };
      const decryptedData = { apiKey: 'secret-key-123' };
      
      const mockVault = {
        read: sandbox.stub().resolves({ data: { data: encryptedData } })
      };
      securityManager.vault = mockVault;

      const decryptStub = sandbox.stub(securityManager, 'decryptSensitiveData')
        .resolves(decryptedData);
      const auditStub = sandbox.stub(securityManager, 'auditSecretAccess')
        .resolves();

      const result = await securityManager.retrieveSecret('consensus/api');

      expect(result).to.deep.equal(decryptedData);
      expect(decryptStub.calledWith(encryptedData)).to.be.true;
      expect(auditStub.calledWith('read', 'consensus/api', 'success')).to.be.true;
    });

    it('should handle secret access failures', async () => {
      const mockVault = {
        read: sandbox.stub().rejects(new Error('Access denied'))
      };
      securityManager.vault = mockVault;

      const auditStub = sandbox.stub(securityManager, 'auditSecretAccess')
        .resolves();

      try {
        await securityManager.retrieveSecret('consensus/forbidden');
        expect.fail('Should have thrown error');
      } catch (error) {
        expect(error.message).to.equal('Access denied');
        expect(auditStub.calledWith('read', 'consensus/forbidden', 'failure')).to.be.true;
      }
    });
  });

  describe('SIEM Integration', () => {
    it('should send events to SIEM', async () => {
      const axiosStub = sandbox.stub(require('axios'), 'post').resolves();
      
      await securityManager.sendToSIEM('security-alert', {
        type: 'unauthorized-access',
        severity: 'HIGH'
      });

      expect(axiosStub.calledOnce).to.be.true;
      expect(axiosStub.firstCall.args[0]).to.include('/services/collector/event');
      expect(axiosStub.firstCall.args[1].event.type).to.equal('security-alert');
    });

    it('should enrich events with context', async () => {
      const axiosStub = sandbox.stub(require('axios'), 'post').resolves();
      
      await securityManager.sendToSIEM('test-event', { data: 'test' });

      const sentEvent = axiosStub.firstCall.args[1].event;
      expect(sentEvent.environment).to.exist;
      expect(sentEvent.service).to.equal('consensus-system');
      expect(sentEvent.correlationId).to.exist;
      expect(sentEvent.severity).to.exist;
    });
  });

  describe('Compliance Automation', () => {
    it('should run SOC2 compliance checks', async () => {
      const stubs = {
        checkEncryptionInTransit: sandbox.stub(securityManager, 'checkEncryptionInTransit')
          .resolves({ check: 'encryption-in-transit', passed: true }),
        checkAccessControls: sandbox.stub(securityManager, 'checkAccessControls')
          .resolves({ check: 'access-controls', passed: true }),
        checkAuditLogging: sandbox.stub(securityManager, 'checkAuditLogging')
          .resolves({ check: 'audit-logging', passed: false, reason: 'Missing logs' })
      };

      const results = await securityManager.runComplianceCheck('SOC2');

      expect(results.framework).to.equal('SOC2');
      expect(results.status).to.equal('NON_COMPLIANT');
      expect(results.findings).to.have.length.greaterThan(0);
    });

    it('should run GDPR compliance checks', async () => {
      const stubs = {
        checkPrivacyPolicy: sandbox.stub(securityManager, 'checkPrivacyPolicy')
          .resolves({ check: 'privacy-policy', passed: true }),
        checkRightToErasure: sandbox.stub(securityManager, 'checkRightToErasure')
          .resolves({ check: 'right-to-erasure', passed: true })
      };

      const results = await securityManager.runComplianceCheck('GDPR');

      expect(results.framework).to.equal('GDPR');
      expect(results.checks).to.be.an('array');
    });

    it('should collect compliance evidence', async () => {
      securityManager.config.compliance.evidenceCollection = true;
      const evidenceStub = sandbox.stub(securityManager, 'collectComplianceEvidence')
        .resolves();

      await securityManager.runComplianceCheck('SOC2');

      expect(evidenceStub.calledOnce).to.be.true;
    });
  });

  describe('Threat Detection', () => {
    it('should detect authentication anomalies', async () => {
      const anomalies = [{
        type: 'brute-force',
        userId: 'user123',
        attempts: 50,
        timeWindow: 300000
      }];

      sandbox.stub(securityManager, 'detectAuthenticationAnomalies')
        .resolves(anomalies);

      const emitSpy = sandbox.spy(securityManager, 'emit');
      await securityManager.detectThreats();

      expect(emitSpy.calledWith('threat-detected')).to.be.true;
    });

    it('should respond to critical threats automatically', async () => {
      const threat = {
        id: 'threat-123',
        severity: 'CRITICAL',
        type: 'container-escape'
      };

      sandbox.stub(securityManager, 'detectContainerAnomalies')
        .resolves([threat]);
      const responseStub = sandbox.stub(securityManager, 'respondToThreat')
        .resolves();

      await securityManager.detectThreats();

      expect(responseStub.calledWith(threat)).to.be.true;
    });
  });

  describe('Incident Response', () => {
    it('should create and handle incidents', async () => {
      const threat = {
        id: 'threat-123',
        severity: 'CRITICAL',
        type: 'data-exfiltration'
      };

      const incident = await securityManager.createIncident(threat);

      expect(incident.id).to.exist;
      expect(incident.threat).to.deep.equal(threat);
      expect(incident.status).to.equal('OPEN');
      expect(incident.severity).to.equal('CRITICAL');
    });

    it('should execute incident response phases', async () => {
      const incident = {
        id: 'incident-123',
        threat: { type: 'malware' },
        status: 'OPEN'
      };

      const phaseStubs = {
        contain: sandbox.stub(securityManager, 'containThreat').resolves('contained'),
        investigate: sandbox.stub(securityManager, 'investigateIncident').resolves('investigated'),
        eradicate: sandbox.stub(securityManager, 'eradicateThreat').resolves('eradicated'),
        recover: sandbox.stub(securityManager, 'recoverFromIncident').resolves('recovered'),
        lessons: sandbox.stub(securityManager, 'documentLessonsLearned').resolves('documented')
      };

      await securityManager.executeIncidentResponse(incident);

      Object.values(phaseStubs).forEach(stub => {
        expect(stub.calledOnce).to.be.true;
      });
      expect(incident.status).to.equal('RESOLVED');
    });
  });
});

// Helper functions
function generateTestCertificate(options = {}) {
  const cert = {
    subject: 'CN=test.consensus.local',
    issuer: 'CN=Consensus CA',
    serialNumber: '123456',
    validFrom: options.expired ? new Date(Date.now() - 86400000) : new Date(),
    validTo: options.expired ? new Date(Date.now() - 3600000) : new Date(Date.now() + 86400000)
  };
  
  return Buffer.from(JSON.stringify(cert)).toString('base64');
}

function generateTestJWT(claims = {}) {
  const header = { alg: 'RS256', typ: 'JWT', kid: 'test-key' };
  const payload = {
    sub: 'test-user',
    iss: 'https://test-auth.local',
    aud: 'test-client',
    exp: Math.floor(Date.now() / 1000) + 3600,
    iat: Math.floor(Date.now() / 1000),
    ...claims
  };
  
  if (claims.invalidSignature) {
    return 'invalid.jwt.token';
  }
  
  const encodedHeader = Buffer.from(JSON.stringify(header)).toString('base64url');
  const encodedPayload = Buffer.from(JSON.stringify(payload)).toString('base64url');
  const signature = 'test-signature';
  
  return `${encodedHeader}.${encodedPayload}.${signature}`;
}